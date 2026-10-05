"""Lat-lon C-grid FV ocean model with split-explicit barotropic/baroclinic stepping.

Arakawa C-grid staggering eliminates the 2*dx checkerboard null space
that plagues collocated A-grid formulations at moderate resolutions.

  u -> lon interfaces  (n_lat, n_lon+1, nlev)
  v -> lat interfaces  (n_lat+1, n_lon, nlev)
  eta, T, S -> cell centers (n_lat, n_lon [, nlev])

Uses the same split-explicit scheme as the A-grid LatLonOceanModel:
1. Compute 3D baroclinic tendencies (slow mode)
2. Update tracers (T, S) and 3D velocity with slow tendency
3. Run barotropic substeps (forward-backward) for free surface
4. Apply conservation fixers

Public API: state_new = model.step(state, dt)
"""

from __future__ import annotations

import os
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field
from legoesm.core.precision import cast_pytree
from legoesm.grids.latlon import (
    LatLonGrid,
    compute_v_face_coords,
    ensure_geometry,
)
from legoesm.grids.polar_filter import (
    compute_polar_filter_mask,
    fourier_filter,
    fourier_filter_3d,
)
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    OceanPartialCellCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
    diagnose_w_from_flux_div,
    flux_form_vertical_tracer_advection,
    flux_form_vertical_tracer_advection_tvd,
    flux_form_vertical_tracer_advection_centered,
    nemo_qco_live_face_geometry_cgrid,
    nemo_qco_live_face_geometry_from_operands,
    nemo_qco_live_t_thickness,
    nemo_qco_card_mesh_operands,
    nemo_qco_mesh_operands,
    nemo_up3_vertical_momentum_advection,
    nemo_wicker_aimp_partition_transport,
)
from legoesm.ocean.state import (
    LatLonCGridOceanState,
    LatLonCGridOceanConfig,
    constants_equal,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
    compute_frozen_geom_density,
    nemo_qco_kmm_velocity_cycle,
    nemo_qco_wzv_operands,
    nemo_rk3_after_ssh_is_carried,
    interp_to_v_points,
    interp_to_v_points_multi,
    centered_cell_to_uface,
    upwind_to_u_points,
    upwind_to_v_points,
    tvd_to_u_points,
    tvd_to_v_points,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks,
    compute_face_masks_3d,
    divergence_cgrid,
    interp_cell_to_uface,
    interp_cell_to_vface,
    min_cell_to_uface,
    min_cell_to_vface,
)
from legoesm.ocean.dynamics.barotropic_common import (
    nemo_literal_after_level_reconcile,
    nemo_reference_depth_reciprocal,
    rk3_stage_barotropic_correction,
    validate_after_reconcile,
)
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    barotropic_substeps_latlon_cgrid,
)
from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
    barotropic_implicit_latlon_cgrid,
)
from legoesm.ocean.dynamics.ocean_tendency_common import (
    ab2_blend,
    depth_average_to_faces,
    depth_mean,
)
from legoesm.ocean.freshwater import (
    freshwater_eta_tendency,
    net_freshwater_flux,
    virtual_salt_flux,
)
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ocean.physics.combined import make_ocean_physics
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    compute_eke_step_kappa,
    compute_geometric_step_kappa,
    geometric_barotropic_production,
    compute_realized_gm_skew_conversion,
    compute_realized_signed_conversions,
    eke_horizontal_transport,
    eke_3d_horizontal_transport,
    eke_3d_vertical_diffusion,
    gm_redi_tracer_tendency_latlon,
    validate_redi_coefficient,
    gm_redi_density_and_jacobian,
    harmonic_lateral_kediss_eke_source,
    compute_isoneutral_K33_latlon,
)
from legoesm.ocean.physics.lateral_mixing.eke import eke_apply_local_source
# Public symbol from the blessed shared "_gm_redi_common" module pattern (see
# tests/test_no_private_cross_imports.py): the SAME Hallberg f_res definition
# the GM/Redi tracer tendencies apply to kappa_GM, reused for the EKE-budget
# production coupling (codex MED-3 r2) — never re-derived.
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    gm_resolution_factor,
)
from legoesm.ocean.advection_som import som_advect_tracers
from legoesm.ocean.conservation import ocean_conservation_fixer


# ---------------------------------------------------------------------------
# store_mass_flux carry (#1442)
# ---------------------------------------------------------------------------
# Units of the stored tracer-advecting mass flux: thickness x velocity.  NOT
# the "m/s" of the u/v Fields whose dims/staggering these inherit (codex
# YELLOW 7 -- a mislabelled Field silently mis-scales any CF/netCDF export).
MASS_FLUX_UNITS = "m^2/s"
# The vertical partner is a VELOCITY at layer interfaces, not a thickness-
# weighted face flux -- the continuity operator already divided by thickness.
MASS_FLUX_W_UNITS = "m/s"
MASS_FLUX_W_DIMS = ("lat", "lon", "level_interface")
MASS_FLUX_SLOTS = ("mass_flux_u", "mass_flux_v", "mass_flux_w")


def mass_flux_fields(u_field, v_field, w_field, mfu_data, mfv_data, mfw_data):
    """Wrap the tracer-advecting mass fluxes as the three ``mass_flux_*`` Fields.

    THE SINGLE CONSTRUCTOR for these Fields (#1442).  Both the hot-path capture
    in ``_step_impl`` and the ``seed_scan_carry`` / SPMD pre-seed go through
    here, so the seeded carry's metadata is IDENTICAL to what the step writes.
    That identity is load-bearing, not cosmetic: ``Field`` carries its
    ``name``/``dims``/``units``/``staggering`` as pytree AUX DATA, so a seed
    that differs in ANY of them is a different treedef and ``lax.scan`` rejects
    the carry (the same trap the leapfrog ``u_before`` seeding comment records).

    ``u_field``/``v_field`` donate dims and staggering (identical face layout);
    only ``name`` and ``units`` are overridden.  ``w_field`` donates only its
    staggering -- ``mass_flux_w`` lives at layer INTERFACES (nlev+1), one more
    level than ``state.w``, so it carries its own dims.
    """
    return (
        u_field.replace(data=mfu_data, name="mass_flux_u",
                        units=MASS_FLUX_UNITS),
        v_field.replace(data=mfv_data, name="mass_flux_v",
                        units=MASS_FLUX_UNITS),
        w_field.replace(data=mfw_data, name="mass_flux_w",
                        dims=MASS_FLUX_W_DIMS, units=MASS_FLUX_W_UNITS),
    )


def _is_canonical_mass_flux(field, canonical, want_shape, want_dtype) -> bool:
    """True when ``field`` already matches what the step writes, exactly.

    The exit condition for :func:`seed_mass_flux_carry`'s fast path.  It must
    cover EVERYTHING the slow path would have produced, or the fast path is a
    hole -- so it compares against a Field BUILT BY ``mass_flux_fields`` from
    the same donors, using ``Field.tree_flatten``'s own aux tuple rather than
    a hand-listed subset.

    That indirection is the point (codex round-8 RED 2).  The hand-listed
    version checked ``name`` and ``units`` and silently ignored ``dims`` (for
    the u/v pair), ``long_name`` and ``staggering`` -- all of which ARE pytree
    aux data, all of which the step re-derives from the donor field, and any
    one of which is enough to make the next ``lax.scan`` reject the carry.
    Comparing the flattened aux tuple cannot omit a member, and it keeps
    tracking ``Field`` if that class ever gains one.

    SHAPE and DTYPE are compared too -- they are dynamic, not aux, so
    ``tree_flatten`` does not cover them, and a ``lax.scan`` rejects a carry on
    either.  The slow path raises on a shape mismatch and zero-fills at the
    donor's dtype, so a fast path that skipped them would silently accept a
    ``v_lower``-shaped slot in a global state, or an f64 slot in an f32 state
    that the step then re-emits at storage precision (codex round-9 YELLOW 3).
    """
    if field is None:
        return False
    if tuple(jnp.shape(field.data)) != tuple(want_shape):
        return False
    if jnp.asarray(field.data).dtype != want_dtype:
        return False
    return field.tree_flatten()[1] == canonical.tree_flatten()[1]


def seed_mass_flux_carry(state, store_mass_flux: bool):
    """Pre-seed the ``mass_flux_*`` slots to zeros for a constant pytree.

    With ``store_mass_flux`` on, the step turns these slots from ``None`` into
    ``Field``s, which CHANGES THE STATE TREEDEF.  Unseeded that breaks every
    multi-step entry path, silently until it crashes:

    * ``lax.scan`` -- carry-in structure != carry-out structure (the model's
      own ``integrate``/``integrate_scan``, the JRA55 block scans in
      ``run_omip``, the CORE2 ``--scan-block`` in ``omip2_applicator``);
    * a direct jitted ``step`` loop -- the first step retraces and recompiles;
    * the SPMD ``shard_map`` -- ``out_specs`` is derived from the INPUT state,
      so an output leaf with no matching spec is an error.

    Zeros are the correct seed (unlike ``bt_hist``, whose zero reads as a
    meaningful "continuation" state): nothing in the step READS these slots --
    they are written unconditionally every step when the flag is on -- so the
    seed value can never reach the trajectory.

    SHAPE-AGNOSTIC: shapes come from ``state.u``/``state.v``/``state.w``, so
    this is correct for the full-domain state AND for the SPMD ``v_lower``
    carrier (where ``state.v`` already has the n_lat, not n_lat+1, leading
    dim).

    CANONICALIZING, not merely filling (codex round-6 YELLOW 4).  Every slot is
    rebuilt through :func:`mass_flux_fields`, INCLUDING ones that arrive
    already populated: a pair restored from a state built by an earlier commit
    carries the old ``units="m/s"`` metadata, and since Field metadata is
    pytree aux data that alone is a treedef mismatch against what the step
    writes.  Preserving such a Field verbatim -- which the first version of
    this function did -- would reintroduce exactly the scan crash it exists to
    prevent.

    VALUES are RETAINED SUBJECT TO THE DTYPE CONVERSION; metadata is
    normalized.  The cast is deliberate and can be lossy (f64 -> the f32
    storage donor), and a caller CAN observe the narrowed values on the seeded
    carry before the next step overwrites them -- so "preserved" would be too
    strong.  It is done because a slot at a different precision is a
    ``lax.scan`` carry mismatch the moment the step re-emits it at storage
    precision, and these are per-step diagnostics.  A populated slot whose
    SHAPE disagrees with its donor field is a hard error instead -- checked
    BEFORE the cast, so a slot that is wrong in both still raises.

    Idempotent, and a no-op when the flag is off.  The already-canonical fast
    path below is keyed on the METADATA, not merely on "every slot is
    non-None" (codex round-6 YELLOW 4 / round-7 YELLOW 5): the cheap
    non-None test is exactly the shortcut that lets a legacy ``m/s`` pair
    through, so it must not be the exit condition.  Persistent-SPMD host loops
    call this every step and hit the fast path from step 2 on.

    NON-GOAL (explicit, so a later reviewer does not re-open it): this is not
    an adversarial boundary.  The checks here exist to catch DRIFT -- a state
    assembled by an older commit, a v_lower carrier handed to a global path, a
    restart archive written under a previous convention -- on a shared HPC
    filesystem where the realistic threats are stale artefacts and my own
    mistakes.  A hand-forged state that satisfies every check and still lies,
    or a caller that mutates the slots after seeding, is out of scope; the
    step overwrites all three unconditionally on the next call anyway.
    """
    if not store_mass_flux:
        return state
    nlev_i = state.w.data.shape[-1] + 1          # interfaces = nlev + 1
    _shapes = {
        "mass_flux_u": tuple(jnp.shape(state.u.data)),
        "mass_flux_v": tuple(jnp.shape(state.v.data)),
        "mass_flux_w": tuple(jnp.shape(state.w.data))[:-1] + (nlev_i,),
    }
    # Reference Fields built from the SAME donors the step uses.  Only their
    # AUX data is read (the donors' own arrays are passed straight through as
    # placeholders -- no allocation, no device work); shapes are checked
    # separately against ``_shapes``.
    _canon = mass_flux_fields(state.u, state.v, state.w,
                              state.u.data, state.v.data, state.w.data)
    _dtype = jnp.asarray(state.u.data).dtype
    if all(_is_canonical_mass_flux(getattr(state, _n), _c, _shapes[_n], _dtype)
           for _n, _c in zip(MASS_FLUX_SLOTS, _canon)):
        return state

    def _data(slot, donor_shape):
        cur = getattr(state, slot)
        if cur is None:
            return jnp.zeros(donor_shape, dtype=_dtype)
        # SHAPE FIRST, then dtype (codex round-10 RED 1).  The round-9 version
        # normalized dtype and RETURNED before validating the shape, so a slot
        # that was wrong in BOTH was silently narrowed and accepted -- the
        # guard stopped firing for exactly the states it most needed to catch.
        # A wrong shape is a hard error; a wrong dtype is a fixable
        # normalization, so the error has to come first.
        got = tuple(jnp.shape(cur.data))
        if got != tuple(donor_shape):
            raise ValueError(
                f"seed_mass_flux_carry: {slot} has shape {got}, expected "
                f"{tuple(donor_shape)}.  A stored mass flux must match its "
                "u/v/w face layout; a mismatch means the state was assembled "
                "for a different grid or a different (v_lower vs global) "
                "staggering carrier.")
        # Dtype is normalized to the donor's: a slot at a different precision
        # is a lax.scan carry mismatch once the step re-emits it at storage
        # precision (codex round-9 YELLOW 3).
        if jnp.asarray(cur.data).dtype != _dtype:
            return jnp.asarray(cur.data).astype(_dtype)
        return cur.data

    mfu, mfv, mfw = mass_flux_fields(
        state.u, state.v, state.w,
        _data("mass_flux_u", _shapes["mass_flux_u"]),
        _data("mass_flux_v", _shapes["mass_flux_v"]),
        _data("mass_flux_w", _shapes["mass_flux_w"]))
    return state._replace(mass_flux_u=mfu, mass_flux_v=mfv, mass_flux_w=mfw)


# ---------------------------------------------------------------------------
# store_salt_flux carry (gateway exact-salt instrument)
# ---------------------------------------------------------------------------
# Column-integrated advective salt flux: thickness x velocity x salinity,
# summed over levels.  2-D by design -- see the ``salt_flux_u_int`` state
# docstring for the cost/scope decision.
SALT_FLUX_UNITS = "psu m^2/s"
SALT_FLUX_U_DIMS = ("lat", "lon_u")
SALT_FLUX_V_DIMS = ("lat_v", "lon")
SALT_FLUX_SLOTS = ("salt_flux_u_int", "salt_flux_v_int")


def salt_flux_fields(u_field, v_field, sfu_data, sfv_data):
    """Wrap the column-integrated salt fluxes as the two ``salt_flux_*_int``
    Fields.  THE SINGLE CONSTRUCTOR, exactly as ``mass_flux_fields`` is for
    the mass triple: the hot-path capture and every seed go through here so
    the carry treedef (Field metadata is pytree AUX data) cannot drift.

    ``u_field``/``v_field`` donate staggering; the dims are OWN 2-D tuples
    (the donors are 3-D level fields) and units are the salt-flux units.
    """
    return (
        u_field.replace(data=sfu_data, name="salt_flux_u_int",
                        dims=SALT_FLUX_U_DIMS, units=SALT_FLUX_UNITS),
        v_field.replace(data=sfv_data, name="salt_flux_v_int",
                        dims=SALT_FLUX_V_DIMS, units=SALT_FLUX_UNITS),
    )


def seed_salt_flux_carry(state, store_salt_flux: bool):
    """Pre-seed ``salt_flux_u_int``/``salt_flux_v_int`` for a constant pytree.

    The salt sibling of :func:`seed_mass_flux_carry`, for the same reason:
    with the flag on the step turns the slots from ``None`` into ``Field``s,
    which changes the state treedef and breaks every multi-step entry path
    (``lax.scan`` carries, jitted step loops, SPMD ``shard_map`` out_specs).
    Zeros are the correct seed -- nothing reads the slots; the step
    overwrites them unconditionally every step when the flag is on.

    SHAPE-AGNOSTIC via the u/v donors (correct for the global state AND the
    SPMD ``v_lower`` carrier); CANONICALIZING like the mass version (a
    populated slot is rebuilt through :func:`salt_flux_fields`, its dtype
    normalized to the donor's, and a shape mismatch is a hard error --
    checked BEFORE the cast).  Idempotent; no-op when the flag is off.
    The same drift-not-adversary NON-GOAL applies.
    """
    if not store_salt_flux:
        return state
    _shapes = {
        "salt_flux_u_int": tuple(jnp.shape(state.u.data))[:-1],
        "salt_flux_v_int": tuple(jnp.shape(state.v.data))[:-1],
    }
    _canon = salt_flux_fields(state.u, state.v, state.u.data, state.v.data)
    _dtype = jnp.asarray(state.u.data).dtype
    if all(_is_canonical_mass_flux(getattr(state, _n), _c, _shapes[_n], _dtype)
           for _n, _c in zip(SALT_FLUX_SLOTS, _canon)):
        return state

    def _data(slot, want_shape):
        cur = getattr(state, slot)
        if cur is None:
            return jnp.zeros(want_shape, dtype=_dtype)
        got = tuple(jnp.shape(cur.data))
        if got != tuple(want_shape):
            raise ValueError(
                f"seed_salt_flux_carry: {slot} has shape {got}, expected "
                f"{tuple(want_shape)}.  A stored salt flux must match its "
                "u/v face layout; a mismatch means the state was assembled "
                "for a different grid or a different (v_lower vs global) "
                "staggering carrier.")
        if jnp.asarray(cur.data).dtype != _dtype:
            return jnp.asarray(cur.data).astype(_dtype)
        return cur.data

    sfu, sfv = salt_flux_fields(
        state.u, state.v,
        _data("salt_flux_u_int", _shapes["salt_flux_u_int"]),
        _data("salt_flux_v_int", _shapes["salt_flux_v_int"]))
    return state._replace(salt_flux_u_int=sfu, salt_flux_v_int=sfv)


# ---------------------------------------------------------------------------
# Advection flux-divergence helpers (extracted for AB2/RK3 reuse)
# ---------------------------------------------------------------------------

def add_bolus_to_advecting_flux(bolus, mass_flux_u, mass_flux_v,
                                 u_mask_3d, v_mask_3d, grid, z_coord):
    """Add the GM eddy-induced (bolus) transport to the TRACER advecting flux.

    NEMO ``traadv``: the eiv velocity is added to the advecting velocity before
    the (monotone FCT) tracer scheme.  ``bolus = (u_eiv, v_eiv, w_eiv)`` is the
    curl-of-ψ TRANSPORT [m^3/s] on the NEMO east-/north-face-of-cell staggering
    (``nemo_eiv_bolus_transport``).  Returns
    ``(mass_flux_u_tr, mass_flux_v_tr, w_baro_tr)`` — the base momentum/continuity
    fluxes are left untouched (the bolus is tracer-advection only).

    Conservation: the vertical bolus is re-diagnosed from the bolus-augmented
    HORIZONTAL flux via the SAME continuity operator that built ``w_baro``, so the
    augmented advecting field is discretely NON-DIVERGENT in the FCT operators —
    EXACT global tracer conservation AND constancy preservation, independent of
    the horizontal metric.  This IS NEMO's ``zww`` eiv (the continuity integral of
    the horizontal bolus divergence).  The bolus is column-non-divergent (ψ=0 at
    the surface + floor ⇒ column-integrated transport divergence = 0), so the z*
    sigma correction contributes nothing (deta/dt_bolus = 0).
    """
    from legoesm.grids.latlon import ensure_geometry as _ensure_geometry
    u_eiv, v_eiv, _w_eiv = bolus              # [m^3/s], cell-indexed E/N faces
    geom = _ensure_geometry(grid)
    # u/v-face metrics = the SAME e2u/e1v the ψ streamfunction used, so
    # transport/metric recovers the mass-flux (h·u) convention divergence_cgrid
    # consumes (mass_flux · face_width = volume transport).  ψ ∝ metric, so the
    # transport is exactly 0 where the metric is 0 (degenerate pole/boundary
    # v-face) — guard the division against 0/0 (→0, the correct no-flux value).
    e2u = geom.dy_u[:, 1:]                     # (n_lat, n_lon) east face of cell i
    e1v = geom.dx_v[1:, :]                     # (n_lat, n_lon) north face of cell j
    bmfu_c = jnp.where(e2u[:, :, jnp.newaxis] > 0.0,
                       u_eiv / jnp.where(e2u[:, :, jnp.newaxis] > 0.0,
                                         e2u[:, :, jnp.newaxis], 1.0), 0.0)
    bmfv_c = jnp.where(e1v[:, :, jnp.newaxis] > 0.0,
                       v_eiv / jnp.where(e1v[:, :, jnp.newaxis] > 0.0,
                                         e1v[:, :, jnp.newaxis], 1.0), 0.0)
    # Map cell-indexed EAST/NORTH faces -> staggered (n_lon+1)/(n_lat+1) face
    # arrays: face f is the WEST/SOUTH face of cell f, so face f>=1 = east/north
    # face of cell f-1; the boundary face 0 takes the periodic wrap (identically
    # the masked wall value on a closed boundary, where the bolus is 0).
    bolus_mfu = jnp.concatenate([bmfu_c[:, -1:, :], bmfu_c], axis=1)  # (n_lat, n_lon+1, nlev)
    bolus_mfv = jnp.concatenate([bmfv_c[-1:, :, :], bmfv_c], axis=0)  # (n_lat+1, n_lon, nlev)
    # Mask the bolus by the SAME 3-D tracer face masks the base mass flux uses
    # (compute_face_masks_3d) so no bolus flows through a partial-cell wall / dry
    # level.  Without this a below-bathymetry DRY level (garbage near-zero
    # thickness) carries a huge spurious bolus velocity (transport / ~0 h) into
    # the FCT → CFL blow-up; the wet-face masking is byte-identical on a flat
    # bottom (all levels of a wet column active).
    bolus_mfu = bolus_mfu * u_mask_3d
    bolus_mfv = bolus_mfv * v_mask_3d
    mfu_tr = mass_flux_u + bolus_mfu
    mfv_tr = mass_flux_v + bolus_mfv
    flux_div_tr = divergence_cgrid(mfu_tr, mfv_tr, grid)
    # NOTE (#1226): this re-diagnosis DISCARDS any w built by the caller, so a
    # correction applied upstream has no effect while
    # gm_bolus_advection="through_fct" -- and it runs even when kappa_GM = 0.
    # Threading the barotropic thickness tendency in here is WRONG: the bolus is
    # non-divergent, so its column integral is zero and mixing a bolus-inclusive
    # divergence with a non-bolus thickness tendency is inconsistent (tested:
    # constancy error 3.3e-05 -> 2.6e-01). The sigma form below is
    # self-consistent because it derives deta_dt from the SAME divergence.
    w_tr = diagnose_w_from_flux_div(flux_div_tr, z_coord, thickness_weighted=True)
    return mfu_tr, mfv_tr, w_tr


def _compute_advection_flux_div(
    tr: jnp.ndarray,
    tracer_advection: str,
    mass_flux_u: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    w_baro: jnp.ndarray,
    h_k_old: jnp.ndarray,
    h_u_old: jnp.ndarray,
    h_v_old: jnp.ndarray,
    grid,
    dt: float,
    recon_fill_mask: jnp.ndarray | None = None,
    linssh_top_flux: bool = False,
    tr_before: jnp.ndarray | None = None,
    return_h_fluxes: bool = False,
    fct_low_order_predictor: str = "one_step",
    fct_base_thickness=None,
    fct_after_thickness=None,
    fct_implicit_w=None,
    return_fct_activity: bool = False,
):
    """Compute advection flux divergence for a single tracer field.

    ``return_h_fluxes`` (static Python bool): additionally return the
    HORIZONTAL face flux pair ``(tracer_flux_u, tracer_flux_v)`` [tracer
    m^2/s per level] as a 4-tuple ``(div_hut, vert_flux_div, F_u, F_v)`` --
    the arrays whose divergence IS ``div_hut``, i.e. what the flux-form
    update actually moves through the faces (store_salt_flux capture).
    RAISES for the schemes that form fluxes inside their own kernels
    (``ppm_fct``/``fct2``/``dst3_multidim``): exposing a DIFFERENT
    reconstruction there would store a flux the model did not apply
    (refuse-not-ignore).

    ``tr_before`` (leapfrog only): BEFORE-level (Kbb) tracer for the FCT
    monotonicity base (see ``fct_tracer_advection``).  ``None`` on the FE/AB2
    path ⇒ base == ``tr`` ⇒ byte-identical.  Only the FCT schemes consume it
    (the other schemes are single-level and already leapfrog-consistent).

    Returns ``(div_hut, vert_flux_div)`` — horizontal and vertical
    components of the total flux divergence.  Units: [tracer · m/s]
    (thickness-weighted: ``div(h·u·T_face)``).  The caller combines
    them into a flux-form tracer update:

        hT_new = h_old * tr - dt * (div_hut + vert_flux_div)

    Notes
    -----
    When used with AB2: the extrapolated flux divergence
    ``(3/2+eps)*F^n - (1/2+eps)*F^{n-1}`` uses thickness-weighted
    divergences from different time levels.  In variable-thickness
    (z-star) runs this introduces an O(dt) accuracy degradation
    because ``F^{n-1}`` was computed with ``h^{n-1}`` mass fluxes
    but is combined with the current ``h_k_old``.  For constant-
    thickness convergence tests this is exact.

    When used with AB2 + FCT/PPM_FCT: the individual ``F^n`` and
    ``F^{n-1}`` are separately monotone (Zalesak-limited), but their
    AB2 linear combination is NOT guaranteed monotone.  This is a
    known limitation shared with MITgcm.
    """
    # Wall tracer BC (#480): zero-gradient (Neumann) fill the RECONSTRUCTION
    # tracer over the dead cells so the flux-form face reconstructions (esp. the
    # wide WENO stencil) see a flat extension across solid walls / topographic
    # steps instead of the masked cold cell (T=0).  The mass-flux carries the
    # wall masking (zero normal flux), so this conserves the wet-domain tracer
    # and is a strict no-op where there is no land.  ``None`` => disabled.
    # NOTE: the filled ``tr`` feeds BOTH the horizontal AND the vertical
    # reconstruction below; that is safe because neumann_fill_cgrid only
    # overwrites DEAD cells (mask < 0.5) — every WET column is bit-identical, so
    # the vertical flux differs only in dead cells, which the caller's
    # active_3d / mass_flux gating discards.  (If a future caller passes a mask
    # that modifies a wet cell, this invariant must be revisited.)
    if recon_fill_mask is not None:
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            neumann_fill_cgrid,
        )
        tr = neumann_fill_cgrid(tr, recon_fill_mask, grid=grid)
        # Same wall fill for the BEFORE base so its upwind flux + bounds see the
        # identical flat wall extension (dead cells only ⇒ wet columns intact).
        if tr_before is not None:
            tr_before = neumann_fill_cgrid(tr_before, recon_fill_mask, grid=grid)

    if return_h_fluxes and tracer_advection in ("ppm_fct", "fct2",
                                                "dst3_multidim"):
        raise ValueError(
            f"return_h_fluxes: tracer_advection={tracer_advection!r} forms "
            "its limited fluxes inside its own kernel and does not expose "
            "them; a store_salt_flux capture here would not be the applied "
            "flux.  Use an exposed-flux scheme (upwind/tvd/superbee/"
            "centered/ppm/dst3/weno5/weno7).")
    if tracer_advection in ("ppm_fct", "fct2"):
        from legoesm.ocean.advection import fct_tracer_advection
        fct_result = fct_tracer_advection(
            tr, mass_flux_u, mass_flux_v, w_baro, h_k_old, grid, dt,
            high_order="ppm" if tracer_advection == "ppm_fct" else "centred2",
            tracer_before=tr_before,
            # #1226 item 8: same wet mask already threaded as
            # recon_fill_mask (is_active/active_3d) — faithfully masks
            # the NEMO nonosc per-point bound at dry cells (see
            # fct_tracer_advection's active_mask docstring).
            active_mask=recon_fill_mask,
            # key_linssh: thickness is FIXED (the eta volume change is a
            # separate top concentration/dilution flux added after
            # limiting), so the limiter's after-thickness must not move
            # (codex 2026-08-10 on the h_new certification fix).
            fixed_thickness=linssh_top_flux,
            low_order_predictor=fct_low_order_predictor,
            base_thickness=fct_base_thickness,
            after_thickness=fct_after_thickness,
            implicit_w=fct_implicit_w,
            return_limiter_activity=return_fct_activity,
        )
        if return_fct_activity:
            div_hut, vert_flux_div, fct_activity = fct_result
        else:
            div_hut, vert_flux_div = fct_result
    elif tracer_advection == "ppm":
        from legoesm.ocean.advection import (
            ppm_to_u_points, ppm_to_v_points,
            flux_form_vertical_tracer_advection_ppm,
        )
        tr_u = ppm_to_u_points(tr, mass_flux_u)
        tr_v = ppm_to_v_points(tr, mass_flux_v)
        tracer_flux_u = mass_flux_u * tr_u
        tracer_flux_v = mass_flux_v * tr_v
        div_hut = divergence_cgrid(tracer_flux_u, tracer_flux_v, grid)
        vert_flux_div = flux_form_vertical_tracer_advection_ppm(
            tr, w_baro, h_k_old, dt)
    elif tracer_advection == "dst3":
        from legoesm.ocean.advection import (
            dst3_to_u_points, dst3_to_v_points,
            flux_form_vertical_tracer_advection_dst3,
        )
        tr_u = dst3_to_u_points(tr, mass_flux_u, h_u_old, grid, dt)
        tr_v = dst3_to_v_points(tr, mass_flux_v, h_v_old, grid, dt)
        tracer_flux_u = mass_flux_u * tr_u
        tracer_flux_v = mass_flux_v * tr_v
        div_hut = divergence_cgrid(tracer_flux_u, tracer_flux_v, grid)
        vert_flux_div = flux_form_vertical_tracer_advection_dst3(
            tr, w_baro, h_k_old, dt)
    elif tracer_advection == "dst3_multidim":
        from legoesm.ocean.advection import multidim_tracer_advection
        div_hut, vert_flux_div = multidim_tracer_advection(
            tr, mass_flux_u, mass_flux_v, w_baro,
            h_k_old, h_u_old, h_v_old, grid, dt,
        )
    elif tracer_advection in ("weno5", "weno7"):
        from legoesm.ocean.advection import (
            weno5_to_u_points, weno5_to_v_points,
            weno7_to_u_points, weno7_to_v_points,
            flux_form_vertical_tracer_advection_weno5,
            flux_form_vertical_tracer_advection_weno7,
        )
        _u_fn, _v_fn, _vert_fn = {
            "weno5": (weno5_to_u_points, weno5_to_v_points,
                      flux_form_vertical_tracer_advection_weno5),
            "weno7": (weno7_to_u_points, weno7_to_v_points,
                      flux_form_vertical_tracer_advection_weno7),
        }[tracer_advection]
        tr_u = _u_fn(tr, mass_flux_u)
        tr_v = _v_fn(tr, mass_flux_v)
        tracer_flux_u = mass_flux_u * tr_u
        tracer_flux_v = mass_flux_v * tr_v
        div_hut = divergence_cgrid(tracer_flux_u, tracer_flux_v, grid)
        vert_flux_div = _vert_fn(tr, w_baro, h_k_old, dt)
    elif tracer_advection == "centered":
        # Veros ``adv_flux_2nd`` (veros/core/advection.py;
        # enable_superbee_advection=False): UNLIMITED centered 2nd-order
        # tracer flux. The face value is the plain 2-cell average
        #   horizontal: T_face = 0.5*(T[i] + T[i+1])
        #   vertical:   T_face = 0.5*(T[k] + T[k+1])
        # and the flux is F = T_face * (h*u) on the C-grid faces. This
        # REUSES the same flux-form divergence machinery as the TVD /
        # WENO / DST3 paths (build a face value -> mass_flux*tr_face ->
        # divergence_cgrid); "centered" is simply the unlimited face
        # value. The horizontal face values come from the canonical
        # centered cell->face interpolations (``centered_cell_to_uface``,
        # periodic in longitude; ``interp_to_v_points``, the centered
        # cell->v-face interp with the solid-wall / tripolar-fold BC).
        # Wall masking is carried by ``mass_flux_u``/``mass_flux_v``
        # (zero through walls) — the analogue of Veros's maskU/maskV.
        #
        # DISPERSION: centered 2nd order is unlimited, hence dispersive —
        # it can over/undershoot near sharp gradients (non-monotone,
        # locally negative tracers) with zero implicit diapycnal mixing.
        # Used for the Veros-faithful ACC comparison; legoESM's production
        # default stays TVD (Van Leer), which is monotone.
        tr_u = centered_cell_to_uface(tr)
        tr_v = interp_to_v_points(tr, grid)
        tracer_flux_u = mass_flux_u * tr_u
        tracer_flux_v = mass_flux_v * tr_v
        div_hut = divergence_cgrid(tracer_flux_u, tracer_flux_v, grid)
        vert_flux_div = flux_form_vertical_tracer_advection_centered(tr, w_baro)
    elif tracer_advection in ("tvd", "superbee", "upwind"):
        from legoesm.ocean.dynamics._flux_limiters import resolve_tvd_limiter
        if tracer_advection in ("tvd", "superbee"):
            limiter_fn = resolve_tvd_limiter(tracer_advection)
            tr_u = tvd_to_u_points(tr, mass_flux_u, limiter_fn=limiter_fn)
            tr_v = tvd_to_v_points(tr, mass_flux_v, grid=grid, limiter_fn=limiter_fn)
        else:
            tr_u = upwind_to_u_points(tr, mass_flux_u)
            tr_v = upwind_to_v_points(tr, mass_flux_v, grid=grid)
        tracer_flux_u = mass_flux_u * tr_u
        tracer_flux_v = mass_flux_v * tr_v
        div_hut = divergence_cgrid(tracer_flux_u, tracer_flux_v, grid)

        if tracer_advection in ("tvd", "superbee"):
            vert_flux_div = flux_form_vertical_tracer_advection_tvd(
                tr, w_baro, h_k_old, dt,
                limiter_fn=resolve_tvd_limiter(tracer_advection),
            )
        else:
            vert_flux_div = flux_form_vertical_tracer_advection(tr, w_baro)
    else:
        raise ValueError(
            f"Unknown tracer_advection literal {tracer_advection!r}; "
            f"expected one of: upwind, centered, tvd, superbee, ppm, "
            f"ppm_fct, dst3, dst3_multidim, weno5, weno7."
        )

    if return_h_fluxes:
        try:
            _h_flux_pair = (tracer_flux_u, tracer_flux_v)
        except NameError:
            # Structurally unreachable: every scheme that falls through to
            # here either bound the pair above or raised at entry.  Kept as a
            # hard error so a NEW scheme added without exposing its fluxes
            # cannot silently return garbage under the capture flag.
            raise ValueError(
                f"return_h_fluxes: scheme {tracer_advection!r} reached the "
                "return without binding tracer_flux_u/v -- wire its capture "
                "or add it to the refused set.")

    if linssh_top_flux:
        # NEMO key_linssh top-cell concentration/dilution: the surface
        # vertical advective flux is F[0] = w0*T0 (first-order, OUTSIDE any
        # limiter — traadv_fct.F90:413-423 with a zero antidiffusive top
        # flux), instead of the rigid F[0]=0 of the stretching-column z*.
        # vert_flux_div[k] = F[k] - F[k+1], so add w0*T0 to level 0.
        # w_baro[...,0] = deta/dt (fixed-thickness continuity); zero on land.
        vert_flux_div = vert_flux_div.at[..., 0].add(
            w_baro[..., 0] * tr[..., 0])

    if return_fct_activity:
        if tracer_advection not in ("ppm_fct", "fct2"):
            raise ValueError(
                "return_fct_activity requires an FCT tracer scheme")
        return div_hut, vert_flux_div, fct_activity
    if return_h_fluxes:
        return div_hut, vert_flux_div, _h_flux_pair[0], _h_flux_pair[1]
    return div_hut, vert_flux_div


# Schemes whose HORIZONTAL reconstruction is strictly level-independent
# (elementwise limiters + lon rolls + lat pads; no cross-level coupling)
# — the set the pair fast-path below may stack along the level axis.
_LEVEL_SEPARABLE_H_SCHEMES = frozenset(
    {"tvd", "superbee", "upwind", "centered"}
)


def compute_advection_flux_div_pair(
    tr_a: jnp.ndarray,
    tr_b: jnp.ndarray,
    tracer_advection: str,
    mass_flux_u: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    w_baro: jnp.ndarray,
    h_k_old: jnp.ndarray,
    h_u_old: jnp.ndarray,
    h_v_old: jnp.ndarray,
    grid,
    dt: float,
    recon_fill_mask: jnp.ndarray | None = None,
    linssh_top_flux: bool = False,
    tr_a_before: jnp.ndarray | None = None,
    tr_b_before: jnp.ndarray | None = None,
    return_b_h_fluxes: bool = False,
    fct_low_order_predictor: str = "one_step",
    fct_base_thickness=None,
    fct_after_thickness=None,
    fct_implicit_w=None,
    return_a_fct_activity: bool = False,
):
    """Advection flux divergence for TWO tracers (T, S) in one pass.

    ``return_b_h_fluxes`` (static Python bool, store_salt_flux capture):
    additionally return tracer B's HORIZONTAL face flux pair, as a 3-tuple
    ``(pair_a, pair_b, (F_u_b, F_v_b))``.  The capture takes the UNFUSED
    per-tracer path unconditionally: the fused level-stack (opt-in via
    LEGOESM_TRACER_PAIR=1, documented bit-identical and measured SLOWER)
    would need its slice bookkeeping duplicated here for no numeric
    difference.  Default False returns the legacy 2-tuple, byte-identical.

    ``tr_a_before``/``tr_b_before`` (leapfrog only): BEFORE-level (Kbb) tracers
    forwarded to the FCT monotonicity base (see ``fct_tracer_advection``).  FCT
    is NOT a ``_LEVEL_SEPARABLE_H_SCHEMES`` member, so it always takes the
    per-tracer fallback below where the before-levels are threaded through;
    ``None`` ⇒ byte-identical FE/AB2 behaviour.

    The horizontal reconstructions of the production schemes
    (``_LEVEL_SEPARABLE_H_SCHEMES``) are level-independent, so both
    tracers ride ONE reconstruction call on a level-axis stack
    ``[tr_a ‖ tr_b]`` — the AL81 ``t_stack`` precedent: pure
    relabeling, no arithmetic.  This halves the remaining per-step
    tracer N-S exchanges (the halo=2 cell pad inside
    ``tvd_to_v_points`` was one of the few unfused sites left, census
    8459326) and halves the horizontal-reconstruction work.  The
    VERTICAL flux divergence couples levels (it must not see the
    stack seam) and is computed per tracer with the original
    functions.  Every other scheme falls back to two single-tracer
    calls — value-identical, no fast path.

    Returns ``((div_hut_a, vert_a), (div_hut_b, vert_b))`` — each pair
    bit-identical to ``_compute_advection_flux_div`` on that tracer
    (identical elementwise ops applied to the same per-level values).
    """
    # Trace-time opt-in gate (LEGOESM_TRACER_PAIR=1 → level-stacked
    # pair; baked into the compiled graph — flip BEFORE first compile).
    # DEFAULT OFF: the stack halves pad count + reconstruction calls
    # but DOUBLES the reconstruction intermediates, and the same-node
    # A/B (job 8460192) measured np=1 −3% and LL128-np8 −17% with no
    # multi-rank win — the vmix field-batching lesson a third time
    # (stacking ADDS bandwidth where the phase is bandwidth-bound).
    # Kept opt-in: bit-identical (tests/ocean/unit/
    # test_tracer_pair_advection.py) and the trade may flip on GPU.
    if (
        return_b_h_fluxes
        or tracer_advection not in _LEVEL_SEPARABLE_H_SCHEMES
        or os.environ.get("LEGOESM_TRACER_PAIR", "0") != "1"
    ):
        pair_a_result = _compute_advection_flux_div(
            tr_a, tracer_advection, mass_flux_u, mass_flux_v,
            w_baro, h_k_old, h_u_old, h_v_old, grid, dt,
            recon_fill_mask=recon_fill_mask,
            linssh_top_flux=linssh_top_flux,
            tr_before=tr_a_before,
            fct_low_order_predictor=fct_low_order_predictor,
            fct_base_thickness=fct_base_thickness,
            fct_after_thickness=fct_after_thickness,
            fct_implicit_w=fct_implicit_w,
            return_fct_activity=return_a_fct_activity,
        )
        if return_a_fct_activity:
            pair_a = pair_a_result[:2]
            activity_a = pair_a_result[2]
        else:
            pair_a = pair_a_result
        out_b = _compute_advection_flux_div(
            tr_b, tracer_advection, mass_flux_u, mass_flux_v,
            w_baro, h_k_old, h_u_old, h_v_old, grid, dt,
            recon_fill_mask=recon_fill_mask,
            linssh_top_flux=linssh_top_flux,
            tr_before=tr_b_before,
            return_h_fluxes=return_b_h_fluxes,
            fct_low_order_predictor=fct_low_order_predictor,
            fct_base_thickness=fct_base_thickness,
            fct_after_thickness=fct_after_thickness,
            fct_implicit_w=fct_implicit_w,
        )
        if return_b_h_fluxes:
            div_b, vert_b, sf_u, sf_v = out_b
            return pair_a, (div_b, vert_b), (sf_u, sf_v)
        if return_a_fct_activity:
            return pair_a, out_b, activity_a
        return pair_a, out_b

    nlev = tr_a.shape[-1]
    # Wall tracer BC (#480): fill the HORIZONTAL reconstruction tracer over the
    # dead cells (see _compute_advection_flux_div).  Vertical advection (below)
    # is per-column, so it keeps the original tr_a/tr_b.  The mask is a 2D / 3D
    # (singleton-level) mask that broadcasts over the stacked 2*nlev axis, or a
    # full per-level (…,nlev) mask that must be re-stacked to (…,2*nlev).
    trs_h = jnp.concatenate([tr_a, tr_b], axis=-1)
    if recon_fill_mask is not None:
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            neumann_fill_cgrid,
        )
        mask_h = recon_fill_mask
        if mask_h.ndim == trs_h.ndim and mask_h.shape[-1] == nlev:
            mask_h = jnp.concatenate([mask_h, mask_h], axis=-1)
        trs_h = neumann_fill_cgrid(trs_h, mask_h, grid=grid)
    trs = trs_h
    mfu2 = jnp.concatenate([mass_flux_u, mass_flux_u], axis=-1)
    mfv2 = jnp.concatenate([mass_flux_v, mass_flux_v], axis=-1)

    if tracer_advection in ("tvd", "superbee"):
        from legoesm.ocean.dynamics._flux_limiters import resolve_tvd_limiter
        limiter_fn = resolve_tvd_limiter(tracer_advection)
        tr_u2 = tvd_to_u_points(trs, mfu2, limiter_fn=limiter_fn)
        tr_v2 = tvd_to_v_points(trs, mfv2, grid=grid, limiter_fn=limiter_fn)
    elif tracer_advection == "upwind":
        tr_u2 = upwind_to_u_points(trs, mfu2)
        tr_v2 = upwind_to_v_points(trs, mfv2, grid=grid)
    else:  # "centered"
        tr_u2 = centered_cell_to_uface(trs)
        tr_v2 = interp_to_v_points(trs, grid)

    div2 = divergence_cgrid(mfu2 * tr_u2, mfv2 * tr_v2, grid)
    div_a, div_b = div2[..., :nlev], div2[..., nlev:]

    if tracer_advection in ("tvd", "superbee"):
        from legoesm.ocean.dynamics._flux_limiters import resolve_tvd_limiter
        _lim = resolve_tvd_limiter(tracer_advection)
        vert_a = flux_form_vertical_tracer_advection_tvd(
            tr_a, w_baro, h_k_old, dt, limiter_fn=_lim)
        vert_b = flux_form_vertical_tracer_advection_tvd(
            tr_b, w_baro, h_k_old, dt, limiter_fn=_lim)
    elif tracer_advection == "upwind":
        vert_a = flux_form_vertical_tracer_advection(tr_a, w_baro)
        vert_b = flux_form_vertical_tracer_advection(tr_b, w_baro)
    else:  # "centered"
        vert_a = flux_form_vertical_tracer_advection_centered(tr_a, w_baro)
        vert_b = flux_form_vertical_tracer_advection_centered(tr_b, w_baro)

    if linssh_top_flux:
        # NEMO key_linssh top-cell flux F[0]=w0*T0 (see the single-tracer
        # gate) — apply to BOTH tracers of the fused fast path.
        vert_a = vert_a.at[..., 0].add(w_baro[..., 0] * tr_a[..., 0])
        vert_b = vert_b.at[..., 0].add(w_baro[..., 0] * tr_b[..., 0])
    return (div_a, vert_a), (div_b, vert_b)


def _ssp_rk3_tracer_step(
    tr: jnp.ndarray,
    tracer_advection: str,
    mass_flux_u: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    w_baro: jnp.ndarray,
    h_k_old: jnp.ndarray,
    h_k_new: jnp.ndarray,
    h_u_old: jnp.ndarray,
    h_v_old: jnp.ndarray,
    grid,
    dt: float,
    active_3d: jnp.ndarray,
    linssh_top_flux: bool = False,
) -> jnp.ndarray:
    """RK3 flux-form tracer advection step (Butcher-tableau form).

    Three-stage Runge-Kutta with the same coefficients as SSP-RK3
    (Shu-Osher 1988), but applied via the effective tendency

        F_eff = F0/6 + F1/6 + 2*F2/3

    in flux form for **exact conservation**:

        h_new * T_new = h_old * T - dt * F_eff

    Trade-off: the Shu-Osher convex-combination form preserves the
    strong stability (monotonicity/TVD) property of the forward Euler
    operator, but does not conserve mass exactly when h changes.
    This Butcher form conserves mass exactly but does NOT preserve
    SSP — with nonlinear limiters (TVD, WENO, FCT) new extrema may
    appear that would not appear under the true Shu-Osher form.

    Intermediate stages use ``h_k_old`` (exact for constant-thickness
    convergence tests, O(dt) approximation for the full model where
    thickness changes are from the barotropic step).  The velocity
    field is frozen across all three stages (correct for the
    operator-split advection sub-problem where velocity comes from
    the barotropic solver).
    """

    def _flux_div(tr_val):
        dh, dv = _compute_advection_flux_div(
            tr_val, tracer_advection, mass_flux_u, mass_flux_v, w_baro,
            h_k_old, h_u_old, h_v_old, grid, dt,
            linssh_top_flux=linssh_top_flux,
        )
        return dh + dv

    h_safe = jnp.maximum(h_k_old, 1e-10)

    # Stage 1
    fd0 = _flux_div(tr)
    tr1 = (h_k_old * tr - dt * fd0) / h_safe
    tr1 = jnp.where(active_3d > 0.5, tr1, tr)

    # Stage 2
    fd1 = _flux_div(tr1)
    tr1_adv = (h_k_old * tr1 - dt * fd1) / h_safe
    tr1_adv = jnp.where(active_3d > 0.5, tr1_adv, tr)
    tr2 = 0.75 * tr + 0.25 * tr1_adv

    # Stage 3 — conservative final update via effective flux
    fd2 = _flux_div(tr2)
    F_eff = (1.0 / 6.0) * fd0 + (1.0 / 6.0) * fd1 + (2.0 / 3.0) * fd2
    hT_new = h_k_old * tr - dt * F_eff
    tr_new = hT_new / jnp.maximum(h_k_new, 1e-10)
    tr_new = jnp.where(active_3d > 0.5, tr_new, tr)

    return tr_new


def _ssp_rk3_tracer_pair_step(
    tr_a: jnp.ndarray,
    tr_b: jnp.ndarray,
    tracer_advection: str,
    mass_flux_u: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    w_baro: jnp.ndarray,
    h_k_old: jnp.ndarray,
    h_k_new: jnp.ndarray,
    h_u_old: jnp.ndarray,
    h_v_old: jnp.ndarray,
    grid,
    dt: float,
    active_3d: jnp.ndarray,
    recon_fill_mask: jnp.ndarray | None = None,
    linssh_top_flux: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """RK3 tracer step for the (T, S) pair — one fused flux-div per stage.

    Per-tracer arithmetic is kept statement-identical to
    :func:`_ssp_rk3_tracer_step`; the only change is that each stage's
    two flux divergences come from ONE
    :func:`compute_advection_flux_div_pair` call (3 fused horizontal
    reconstructions + pads per step instead of 6).

    ``recon_fill_mask`` (#480) is forwarded to every stage's reconstruction
    so each RK3 sub-stage sees the zero-gradient wall fill; the flux-form
    stage updates / land-gating still use the ORIGINAL (un-filled) tracer.
    """

    def _flux_div_pair(a_val, b_val):
        (dh_a, dv_a), (dh_b, dv_b) = compute_advection_flux_div_pair(
            a_val, b_val, tracer_advection, mass_flux_u, mass_flux_v,
            w_baro, h_k_old, h_u_old, h_v_old, grid, dt,
            recon_fill_mask=recon_fill_mask,
            linssh_top_flux=linssh_top_flux,
        )
        return dh_a + dv_a, dh_b + dv_b

    h_safe = jnp.maximum(h_k_old, 1e-10)

    # Stage 1
    fd0_a, fd0_b = _flux_div_pair(tr_a, tr_b)
    a1 = (h_k_old * tr_a - dt * fd0_a) / h_safe
    a1 = jnp.where(active_3d > 0.5, a1, tr_a)
    b1 = (h_k_old * tr_b - dt * fd0_b) / h_safe
    b1 = jnp.where(active_3d > 0.5, b1, tr_b)

    # Stage 2
    fd1_a, fd1_b = _flux_div_pair(a1, b1)
    a1_adv = (h_k_old * a1 - dt * fd1_a) / h_safe
    a1_adv = jnp.where(active_3d > 0.5, a1_adv, tr_a)
    a2 = 0.75 * tr_a + 0.25 * a1_adv
    b1_adv = (h_k_old * b1 - dt * fd1_b) / h_safe
    b1_adv = jnp.where(active_3d > 0.5, b1_adv, tr_b)
    b2 = 0.75 * tr_b + 0.25 * b1_adv

    # Stage 3 — conservative final update via effective flux
    fd2_a, fd2_b = _flux_div_pair(a2, b2)
    h_new_safe = jnp.maximum(h_k_new, 1e-10)
    F_eff_a = (1.0 / 6.0) * fd0_a + (1.0 / 6.0) * fd1_a + (2.0 / 3.0) * fd2_a
    a_new = (h_k_old * tr_a - dt * F_eff_a) / h_new_safe
    a_new = jnp.where(active_3d > 0.5, a_new, tr_a)
    F_eff_b = (1.0 / 6.0) * fd0_b + (1.0 / 6.0) * fd1_b + (2.0 / 3.0) * fd2_b
    b_new = (h_k_old * tr_b - dt * F_eff_b) / h_new_safe
    b_new = jnp.where(active_3d > 0.5, b_new, tr_b)

    return a_new, b_new


class _NEMOVerticalSolveTestInput(NamedTuple):
    """Fixed-shape private input for production-compiled solve probes.

    ``replace`` is a dynamic six-element boolean array.  Disabled, identity,
    directed, and planted probes therefore share one JIT input structure and
    one traced graph; only leaf values differ.  No model configuration can
    construct or select this diagnostic input.
    """

    heat_K: object
    viscosity_K: object
    formed_K: object
    tracer_e3w: object
    tracer_e3t: object
    temperature_content: object
    replace: object


# Legal values of the private ``expose_stage1_momentum_rhs_split`` hook: the
# empty default, the two frames round 204 added, and the two halves round 205
# splits the removed advection content into.  A closed set so an unknown
# string cannot select an exposure silently.
_STAGE1_SPLIT_ARMS = ("", "pre_advection", "completed",
                      "advection_horizontal", "advection_vertical",
                      "advection_zub_increment")
# Legal values of ``momentum_transport_stage1_operand``: the production path
# and the two one-variable swaps of stprk3_stg.f90:270's operands.  Round 206
# landed NEMO's depth, so the depth arm is now the LEGACY one.
_STAGE1_TRANSPORT_OPERAND_ARMS = ("", "prognostic_mean",
                                  "legacy_min_rule_depth")


class _NEMOWSRK3TestHooks(NamedTuple):
    """Private causal controls; never part of a constructible model config."""

    # Return the 50-substep dynspg_ts trace from the production-jitted step.
    # This is deliberately a static private hook: fidelity gates must not call
    # ``_step_impl`` eagerly to obtain internals, because eager execution does
    # not certify the compiled production arithmetic.
    expose_barotropic_substeps: bool = False
    # Return every already-materialized operand from the one compiled WS-RK3
    # stage program.  Private WRITE-only round-51 instrument; no constructible
    # model configuration can select it.
    expose_live_stage_operands: bool = False
    # Private controls for the frozen barotropic forcing walk.  Incoming arrays
    # land before dyn_cor_2D's equivalent; final arrays land after it.  A
    # callable final value is a WRITE-only operand observer instead.  None is
    # constructible as a model configuration selector.
    slow_forcing_incoming_override: object = None
    barotropic_slow_forcing_override: object = None
    # Substitute the two frozen positive face drag-rate arrays at the external
    # solver boundary.  Private fidelity measurement only: NEMO has no
    # selector for these coefficients and no public config can reach this.
    barotropic_drag_rate_override: object = None
    # Private round-197 per-substep discriminator for the compiled barotropic
    # sub-time-step loop: a pair of (n_substeps, ...) stacks that replace the
    # 2-D Coriolis trend (NEMO dyn_cor_2D, dynspg_ts.f90:503) at the substep
    # the scan is on, keyed by the scan's own substep index.  ``None`` is the
    # production value and is resolved at trace time, so the unset hook leaves
    # the jaxpr unchanged; no constructible model configuration can select it.
    barotropic_substep_coriolis_override: object = None
    # Same instrument for the substep surface-pressure gradient
    # (dynspg_ts.f90:498), so the two operands of the velocity update at
    # dynspg_ts.f90:535 can be substituted one at a time.
    barotropic_substep_pgf_override: object = None
    # WRITE-only developed-state observer for the completed three-dimensional
    # momentum RHS before its depth reduction.  Kept separate from the final
    # slow-forcing callback so the round-141 gate can prove this minimum
    # materialization passive against the ordinary production step.
    slow_forcing_rhs_observer: object = None
    slow_forcing_rhs_observer_face: str = ""
    # WRITE-only round-5 (VORTEX) observer for the PER-TERM decomposition of
    # that same completed right-hand side.  It is handed the
    # ``MomentumTendencyDiagnostics`` the very same ``tendencies`` call
    # already builds -- with the stage face thicknesses, the lateral-diffusion
    # thickness operands and the continuity clock this step supplies -- so the
    # rows are a decomposition of the array the observer above reports and not
    # a second opinion obtained from a different call.  None is the production
    # value and no model configuration can select it.
    slow_forcing_rhs_term_observer: object = None
    # Private round-142 directed discriminator: replace only the owned native
    # faces of the completed 3-D momentum RHS before its depth reduction.
    # None leaves the production program unchanged; this is not configurable.
    slow_forcing_rhs_override: object = None
    # Private round-143 downstream discriminators.  The depth override lands
    # before the live wind/drag arithmetic.  The drag override replaces that
    # completed boundary and then applies the already-computed wind once.
    # Both preserve unowned faces and are absent from the public config.
    slow_forcing_depth_override: object = None
    slow_forcing_drag_override: object = None
    # Private round-144 discriminator for the compiled wind statement.  Each
    # tuple member independently replaces density reciprocal, face stresses,
    # or live inverse depths; None retains the production operand.
    slow_forcing_wind_operand_override: object = None
    # Substitute NEMO's six raw b/bb arrays at the barotropic loop entry while
    # leaving legoESM's carried state untouched.  This covers both a cold-start
    # ll_init frame and a continuation frame; private fidelity measurement only.
    barotropic_raw_history_override: object = None
    # Stage-twin driver only: replace the completed external solve handoff
    # (eta, uu_b, vv_b, Hu_avg, Hv_avg) with a recorded NEMO handoff before
    # any RK3 stage consumes it.  This is private instrumentation, never a
    # constructible configuration choice.
    stage_barotropic_output_override: object = None
    # Stage-twin driver only: replace one stage's Kmm entry bundle
    # (stage number, u, v, T, S, eta) at the pointer-swap boundary.  Stage 1
    # is supplied by the prognostic input state; valid override stages are 2
    # and 3.  The ordinary stage still runs through the shared implementation.
    stage_entry_override: object = None
    # Private companion for the 19-frame OVERFLOW causal gate. ``None``
    # preserves the production predicate; ``False`` restores the legacy
    # velocity-form update while the trace still crosses the compiled return
    # pytree. No public configuration can select this test-only arm.
    barotropic_flux_form_update_override: object = None

    # Ablates the per-stage external-mode REPLACEMENT inside the one WS stage
    # ladder (stprk3_stg.F90:433-446): each stage then keeps its own depth
    # mean.  It does not select a second ladder -- there is only one.
    stage_barotropic_correction: bool = True
    momentum_transport_reconcile: bool = True
    kmm_tracer_transports: bool = True
    two_step_fct_predictor: bool = True
    disable_bbl: bool = False
    # Private faithful-but-worse control: restore the pre-census continuous
    # partial-centroid BBL geometry.  Public NEMO WS-RK3 configurations use
    # tra_bbl_init's reference-depth mask and raw face e3*_0 unconditionally;
    # NEMO has no switch between these geometries (trabbl.F90:507-533).
    legacy_bbl_partial_geometry: bool = False
    # Private post-adjudication controls.  The NEMO WS-RK3 identity consumes
    # one raw e3w_0 ladder in both wAimp and ZDF; these restore the former
    # partial-cell midpoint at one consumer for causal/reproduction tests.
    legacy_aimp_midpoint_w_metric: bool = False
    legacy_zdf_midpoint_w_metric: bool = False
    # Private ablation of dynspg_ts's flux-form primary transport average.
    # Public NEMO RK3 configurations always keep this true.
    primary_transport_average: bool = True
    # Experimental operator-class ablation for the OVERFLOW stability probe.
    # NEMO has no such switch: public RK3 configurations always retain their
    # stage-3 vertical tracer transport.  This hook only zeroes the w transport
    # handed to the tracer flux path; horizontal transport is unchanged.
    disable_tracer_vertical_transport: bool = False
    # Experimental OVERFLOW localization only.  NEMO's resolved
    # ln_zad_Aimp is true, so public cards always retain adaptive momentum;
    # this hook asks whether legoESM's current post-program approximation is
    # itself an instability owner before the source-exact package is written.
    disable_adaptive_implicit_momentum: bool = False
    # Return a momentum stage's instantaneous velocity in the prognostic u/v
    # slots after the full step has run.  Private fidelity instrumentation only;
    # zero leaves the returned state untouched.
    expose_momentum_stage: int = 0
    # WRITE-only companion for the round-11 composition walk: expose stage-2
    # Kaa immediately after the selected stprk3_stg update and before the
    # reference-depth barotropic-mean replacement.  The ordinary step still
    # completes before the diagnostic substitutes the returned u/v slots.
    expose_stage2_raw_momentum: bool = False
    # Restores the pre-fix LIVE-weighted stage depth mean.  NEMO removes a
    # REFERENCE-weighted one -- ``zub = uu_b(Kaa) - SUM(e3u_0*uu(:,Kaa))
    # * r1_hu_0`` (stprk3_stg.F90:440) with ``hu_0 = SUM(e3u_0*umask)``
    # (domain.F90:145) -- and NEMO has no switch for it; this is a
    # one-variable gate ablation only.
    legacy_live_stage_mean_weights: bool = False
    # Harness-only Arm A: restore the pre-fix Kbb EOS/HPG operands in stages
    # 2/3. Public NEMO WS-RK3 always uses live Kmm T/S/ssh.
    freeze_stage_hpg_operands: bool = False
    # Harness-only SPLIT of the freeze arm (review round): freeze only the
    # stage T/S (live eta) or only the stage eta (live T/S) handed to the
    # stage-2/3 eos+hpg call.  Together they equal freeze_stage_hpg_operands.
    freeze_stage_hpg_tracers: bool = False
    freeze_stage_hpg_eta: bool = False
    # Harness-only Arm B: omit dyn_adv_up3's vertical flux from EVERY stage
    # RHS.  Public NEMO WS-RK3 always calls the full dyn_adv package each
    # stage (stprk3_stg.F90:315,331-334).
    omit_stage_vertical_up3: bool = False
    # Private operand diagnostic: return the stage-advanced tracer carried by
    # Kmm after stage 1 or 2. Zero leaves the returned tracer untouched.
    expose_tracer_stage: int = 0
    # WRITE-only source-order diagnostic for the stage-1 tracer accumulator.
    # ``after_advection`` returns ts(Krhs) after tra_adv; ``after_sbc`` returns
    # it after tra_sbc_RK3.  The ordinary compiled step still completes before
    # T/S are substituted in the returned state.  No card can select this.
    expose_tracer_stage1_boundary: str = ""
    # Harness-only ablation of NEMO's qco stage face thickness: restore the
    # pre-fix min-of-stretched-T-thicknesses rule in the stage transport.
    # NEMO has no such switch (e3u(Kmm) is a macro), so public WS-RK3 cards
    # always use e3u_0*(1+r3u).
    legacy_stage_min_face_thickness: bool = False
    # Harness-only ablation of the qco stage velocity weighting
    # (stprk3_stg.F90:373-378 and dynzdf.F90's key_qco branch): restore the
    # unweighted ``u_raw = u_Kbb + stage_dt*RHS``.  Public WS-RK3 always
    # carries the (1+r3u(Kbb)) / (1+r3u(Kmm)) / (1+r3u(Kaa)) factors.
    omit_stage_qco_factor: bool = False
    # One-variable ablation for the vector-invariant half of the collapsed
    # WS-RK3 identity. NEMO's live ``ln_dynadv_vec`` branch advances stages
    # 1/2 directly on velocity (stprk3_stg.F90:365-369); this restores the
    # former, dead flux-form QCO weighting at :370-386 for the causal gate.
    legacy_vector_stage_qco_weights: bool = False
    # One-variable ablation of the Kmm face thickness in the vector ENE
    # transport products. Public NEMO WS-RK3 uses the same canonical QCO
    # e3u/e3v(Kmm) pair as its stage transport; this restores the former
    # min-of-stretched-T pair for the causal gate only.
    legacy_vector_ene_min_face_thickness: bool = False
    # One-variable ablation of the NEMO-native F-point Coriolis mapping.
    # Public GYRE consumes bridge-carried ff_f at vertex [j+1,i+1]; this
    # restores the generic one-column-west vertex field for the causal gate.
    legacy_ene_vertex_coriolis: bool = False
    # One-variable ablation of the source-associated key_qco tracer stage
    # update. Public NEMO forms ``(qbb*Tbb + dt*qmm*Krhs)/qaa`` directly
    # (stprk3_stg.F90:674-677); this restores the formerly algebraic
    # e3t-weighted content form for the causal gate only.
    legacy_tracer_stage_thickness_association: bool = False
    # One-variable HPG causal arm. Public ``nemo_sco`` uses the literal
    # dynhpg recurrence; this restores the former collapsed p'-gradient
    # association without exposing a constructible scheme selector.
    legacy_hpg_algebraic_association: bool = False
    # Private one-variable ablations for the Round-18 wind walk.  Production
    # preserves a supplied native T-point stress through sbcmod.F90:543-546
    # and materializes stp2d.F90:200-201's written product/add association.
    # These restore the lossy geographic round trip and collapsed division;
    # no constructible model configuration can select either arm.
    legacy_geographic_surface_stress_arm: bool = False
    legacy_barotropic_wind_association: bool = False
    # One-variable ablation of sbcmod.F90:543-546's coastal stress factors.
    # Public NEMO identities always apply them; this private hook restores the
    # former bare face interpolation for movement/scaling evidence only.
    legacy_coastal_surface_stress_factors: bool = False
    # Private ablation of dynspg_ts.F90:629's written continuity association.
    # Production NEMO identities materialize ssh_frc + zhdiv before the
    # rDt_e product; this restores the formerly collapsed algebra only for a
    # one-variable causal measurement.
    legacy_barotropic_continuity_association: bool = False
    # Private one-variable ablation of stprk3_stg.F90:265-278.  Production
    # hands the separately carried uu_b/vv_b value to zub/zvb; this restores
    # the former reduction of the 3-D Kmm field.  No public card selects it.
    legacy_reduced_stage_transport_mean_arm: bool = False
    # Private ablation of traadv.F90:220-226: rederive pFu/pFv from velocity
    # inside WZV instead of consuming the stage transport just materialized.
    legacy_wzv_rederived_transport: bool = False
    # One-variable arm for the S-19 wzv branch: route the WS-RK3 stage
    # cross-level velocity through the SAME nemo_qco_wzv_operands the MLF
    # lane calls (sshwzv.F90:331-336, entered from stprk3_stg.F90:297 with
    # np_transport) instead of the generic diagnose_w_from_flux_div.  NEMO
    # has no such switch -- it is one routine -- so this exists only to
    # measure the two arms against each other on the certified cards.
    literal_stage_wzv: bool = False
    # Private one-variable discriminator for the WS-RK3 stage clock consumed
    # only by sshwzv.F90:334-335.  Production currently passes rn_Dt to all
    # three stages; this arm passes (rn_Dt/3,rn_Dt/2,rn_Dt), matching
    # stprk3_stg.F90:123-124,177-178,221-222.  It is not a public selector.
    source_stage_wzv_clock_arm: bool = False
    # WRITE-only diagnostic companion to expose_tracer_transport_stage: place
    # the raw stage ww in the returned T slot instead of area*ww (pFw).
    expose_tracer_transport_as_ww: bool = False
    # One-variable ablation of the UP3 upwind-selector fix: restore the
    # transport-sign branch choice in the stage horizontal momentum
    # advection.  NEMO dynadv_up3.F90:166-170 has no switch -- the T-point
    # UP3 fluxes always pick the upwind curvature by the sign of the advected
    # velocity pair -- so public WS-RK3 cards never select this.
    legacy_up3_transport_sign_selector: bool = False
    # One-variable ablation of the barotropic loop-entry seed fix: restore the
    # min-of-stretched-cells rescale in the card-mesh ``nemo_literal`` seed
    # instead of NEMO's ``e3u_0*(1+r3u)`` (``dynspg_ts.F90:487`` seeds with
    # ``puu_b(Kmm)``, imposed by ``stprk3_stg.F90:439-446``).  NEMO has no
    # such switch; public cards never select this.
    legacy_seed_min_rule_faces: bool = False
    # One-variable ablation of the resolved GYRE/DINO bottom-drag composition:
    # omit ONLY dynspg_ts's explicit in-substep drag while retaining the
    # production card's implicit dynzdf diagonal and baroclinic split.  NEMO
    # has no switch for this partial program; it is private boundary evidence,
    # never a constructible scheme.
    omit_barotropic_substep_drag: bool = False
    # Private stage-2 EOS/HPG discriminator.  Replace only the (T, S, eta)
    # bundle consumed by the stage-2 call; no production configuration can
    # construct this hook.  NEMO's source order is stprk3_stg.F90:317-336,
    # after the stage-1 tracer update at :452-565.
    stage2_thermodynamic_override: object = None
    # WRITE-only/one-variable pair for the raw stage-2 Krhs gauge.  The
    # override supplies the complete unprojected NEMO RHS immediately before
    # the literal stage update; the exposure publishes the production RHS
    # only after the ordinary step has completed.
    stage2_momentum_rhs_override: object = None
    expose_stage2_momentum_rhs: bool = False
    # WRITE-only stage-1 companions of the pair above.  ``dyn_adv`` is the
    # ONLY momentum statement NEMO runs in stage 1 of the flux-form program
    # (``stprk3_stg.F90:316``, ``IF( .NOT.ln_dynadv_vec ) CALL dyn_adv( ...,
    # zFu, zFv, zFw )``; the vector-invariant arm has already completed its
    # 3-D RHS in ``stp_2D``), and the thickness-weighted explicit update that
    # consumes it is ``stprk3_stg.F90:372-379``.  Scoring either against
    # NEMO's own record needs a stage-1 seam, and the stage-1 arm of
    # ``expose_momentum_operator`` is refused by construction.
    # ``expose_stage1_momentum_rhs`` publishes the completed stage-1 Krhs --
    # NEMO's ``uu(:,:,:,Krhs)`` immediately after that call -- and
    # ``expose_stage1_raw_momentum`` publishes Kaa immediately after the
    # update and BEFORE the barotropic replacement (``:409-421``), the same
    # boundary ``expose_stage2_raw_momentum`` reads one stage later.  Both
    # substitute the returned u/v slots only after the ordinary step has
    # completed, so neither can perturb a later stage; no card constructs
    # them.
    expose_stage1_momentum_rhs: bool = False
    expose_stage1_raw_momentum: bool = False
    # WRITE-only stage-1 SPLIT gauge.  ``stp_2D`` leaves the THREE-dimensional
    # ``Krhs`` with HPG + LDF + COR/MET only: in flux form ``dyn_adv_up3`` is
    # called with ``pUe``/``pVe`` and writes the two-dimensional RHS alone
    # (``stp2d.f90:169-170``; ``dynadv_up3.f90:201,288,349,362-364``), while
    # legoESM's step-level tendency already carries its advection -- so the
    # two codes' pre-stage arrays are NOT like-for-like and round 201 could
    # not attribute the stage-1 error to a statement.  ``"pre_advection"``
    # publishes legoESM's stage-1 right-hand side MINUS the advection content
    # it carries, which IS like-for-like with NEMO's pre-``dyn_adv`` array;
    # ``"completed"`` publishes the completed right-hand side from the SAME
    # evaluation, so a caller can prove that asking ``tendencies()`` for its
    # per-term decomposition perturbed nothing.  Both substitute the returned
    # u/v slots only after the ordinary step has completed, and no card
    # constructs either.
    # Round 205 splits the removed half in two, so the operator NEMO calls
    # at ``stprk3_stg.f90:316`` can be scored part by part against the
    # recorded total: ``"advection_horizontal"`` publishes the flux-form
    # ``-div(transport (x) velocity)`` trend plus the ``zub`` transport
    # increment (``dynadv_up3.f90:174-215``), ``"advection_vertical"`` the
    # vertical UP3 term plus the stage ZAD increment
    # (``dynadv_up3.f90:245-360``).  The two sum to what ``"completed"``
    # minus ``"pre_advection"`` gives.
    expose_stage1_momentum_rhs_split: str = ""
    # Round 205 ONE-VARIABLE CAUSAL PROBE, default off.  NEMO's advective
    # transport subtracts the SEPARATELY PROGNOSTIC depth-mean velocity
    # ``uu_b(:,:,Kmm)`` (``stprk3_stg.f90:270``, the ``n_baro_upd = np_HYB``
    # branch the compiled module's :48 default selects), while legoESM's
    # momentum path RE-REDUCES the three-dimensional velocity to get that
    # mean.  The two are algebraically equal and numerically are not.  With
    # this arm set, the STAGE-1 momentum transport subtracts the prognostic
    # pair instead, so the difference can be scored causally.  Stage 1 only:
    # Kmm = Kbb there, so ``state.uu_b`` IS the Kmm value; stages 2 and 3
    # read a stage-updated pair (``stprk3_stg.f90:433-446``) this probe does
    # not carry, and it refuses to touch them.  ``"legacy_min_rule_depth"``
    # restores the pre-round-206 divisor of the SAME statement: the sum of
    # legoESM's MIN-RULE face thicknesses in place of NEMO's
    # ``hu_0*(1+r3u(Kmm))``.  ``""`` is the production path, which since
    # round 206 (Decision 86) uses NEMO's.
    momentum_transport_stage1_operand: str = ""
    # One-variable companion of ``stage2_momentum_rhs_override``: supply
    # NEMO's own completed stage-1 Krhs (the recorded ``adv_u``/``adv_v``)
    # immediately before the literal stage update, so the update statement
    # (``stprk3_stg.F90:372-379``) and the barotropic replacement
    # (``:409-421``) can be scored as carriers with every other stage input
    # left as legoESM's.  Private diagnostic only; ``None`` keeps the live
    # right-hand side.
    stage1_momentum_rhs_override: object = None
    # WRITE-only stage-3 momentum-RHS gauge.  ``"post_ldf"`` publishes the
    # complete stage-3 Krhs that enters the implicit vertical solve, the
    # operand NEMO hands ``dyn_zdf`` (``stprk3_stg.F90:430``); ``"pre_ldf"``
    # publishes the same RHS with the lateral-mixing term withheld, the
    # operand NEMO hands ``dyn_ldf`` (``stprk3_stg.F90:400``, the ONLY
    # stage-3-only momentum call inside ``CASE ( 3 )`` on this deck).  Those
    # two frames are what separate ``dyn_ldf`` from ``dyn_zdf`` as the owner
    # of the stage-3 divergence.  The ordinary step still completes before
    # either array is substituted into the returned diagnostic state.
    expose_stage3_momentum_rhs: str = ""
    # WRITE-only stage-3 raw-Kaa gauge.  Publish the completed implicit
    # vertical-mixing result immediately before the deferred barotropic
    # replacement (stprk3_stg.F90:430-448).  The ordinary step and replacement
    # still complete before the captured U/V are placed in the returned state.
    expose_stage3_raw_momentum: bool = False
    # WRITE-only observer for the four source-ordered stage-3 dyn_zdf
    # boundaries: explicit update, barotropic subtraction, explicit drag,
    # and implicit solve.  The callback receives eight arrays after the
    # production-jitted step has materialized them; no public card constructs
    # this private fidelity hook.
    zdf_momentum_observer: object = None
    # Private diagnostic controls for WS-RK3 stage-1 source boundaries.
    # Round 119's tuple drives compiled HPG -> LDF -> VOR -> KEG -> ZAD;
    # optional KEG fields are Round-120 arms.  Round 121 may replace only W at
    # the ZAD call boundary.  No card constructs either private control.
    # ``True`` retains the stage-2/3 association discriminator.
    stage1_zad_w_override: object = None
    # WRITE-only callbacks report the W/thickness operands handed to dyn_zad.
    # None is the production value; no card constructs these diagnostics.
    stage1_zad_operand_observer: object = None
    stage2_zad_operand_observer: object = None
    stage3_zad_operand_observer: object = None
    # Rounds 158/194: per-slot substitution at the STAGE-2/3 dyn_zad calls.
    # Each ``(w, h_u, h_v)`` triple uses ``None`` to keep the live operand,
    # so one slot can be replaced by NEMO's recorded stage value while every
    # other input stays legoESM's.  Both feed the existing tendency seam; no
    # card constructs either diagnostic control.
    stage2_zad_operand_override: object = None
    stage3_zad_operand_override: object = None
    # Round 159: private ONE-VARIABLE arm for the stage-2 continuity
    # solve.  NEMO's vector-invariant deck solves it on the RAW stage
    # velocity (``stprk3_stg.f90:360``, velocity indicator) while the
    # production path hands it the barotropically corrected volume
    # transports (transport indicator).  ``True`` selects the velocity
    # indicator at stage 2 only, so the two call forms can be compared
    # with every other operand held.  No card constructs it.
    stage2_wzv_velocity_form: bool = False
    # Round 160 built, round 163 LANDED: NEMO's SECOND per-stage continuity
    # solve (stprk3_stg.f90:360) for the momentum vertical advection, while
    # the tracer transport keeps its own field.  Under Decision 55 (note AT)
    # a NEMO statement that is one-variable and takes a certified row from
    # DEBT to AT-BAR lands even though day 240/360 move outside the noise
    # floor, provided the move is under 1e-3 relative -- see the round-163
    # receipt.  THIS is the TEST-ONLY hook, unchanged in spirit from rounds
    # 159-162: explicit ``True``/``False`` always wins, subject only to
    # ``nemo_stage_momentum_wzv_resolved``, the one-variable way a test
    # reaches either arm on any card.  ``None`` (the default) defers to the
    # CARD's own explicit choice on ``LatLonCGridOceanConfig`` (same field
    # name, a different object) -- GYRE-zco sets it ``True``, ORCA2-zps sets
    # it ``False``, both explicitly, neither inferred from EOS or any other
    # field (round-163 review BLOCKER: an EOS-keyed inference was a hidden
    # coupling between unrelated choices).  A card that resolves the
    # two-solve program without setting the config field raises rather than
    # guessing.
    nemo_stage_momentum_wzv_split: bool | None = None
    # Round 160 WRITE-only exposure: make ``expose_tracer_transport_stage``
    # return the MOMENTUM vertical velocity (slot 11, stprk3_stg.f90:360)
    # instead of the tracer transport's own field, so the two solves can be
    # scored separately against the oracle's recorded array.
    expose_stage_momentum_w: bool = False
    # Round 160 WRITE-only exposure: the stage's own U/V free-surface ratios
    # r3u(Kmm)/r3v(Kmm), the operand divhor.f90:126-130 rebuilds each face
    # transport from.  Stage 1, 2 or 3; 0 is off.
    expose_stage_face_r3: int = 0
    # Round 160 private ONE-VARIABLE arm for the stage-2 momentum solve's
    # quasi-Eulerian stretching term: NEMO's stage clock rDt = rn_Dt/2
    # (stprk3_stg.f90:221-222) with the matching HALF after-level
    # (stprk3_stg.f90:254).  The pair is inseparable -- moving one without the
    # other is a factor of two, not a control -- so one flag selects both.
    stage2_momentum_wzv_clock_pair: bool = False
    nemo_stage_rhs_accumulation_order_arm: object = False
    # WRITE-only transport exposure for the ordered tracer boundary walk.
    # A nonzero stage stores NEMO's metric zFu/zFv/zFw triplet in u/v/T after
    # the ordinary step; it cannot affect a later stage or public execution.
    expose_tracer_transport_stage: int = 0
    # WRITE-only adjudication seam for ORCA2 Phase 2l.  The optional external
    # endpoint tuple is ``(eta_after, Hu_avg, Hv_avg)`` and replaces only the
    # already-registered ORACLE_SUPPLIED external-mode result after the normal
    # compiled solve.  ``expose_stage1_wzv`` then returns the actual ``ww``
    # produced by the one production _g0 transport path, after the ordinary
    # step completes.  Neither field is constructible from a public card.
    external_mode_result_override: object = None
    expose_stage1_wzv: bool = False
    # WRITE-only stage-1 horizontal-transport operand exposure. ``thickness``
    # returns e3u/e3v(Kmm); ``corrected_velocity`` returns uu+zub*umask and
    # vv+zvb*vmask; ``transport_average`` broadcasts un_adv/vn_adv.  The
    # arrays replace returned u/v only after the full step.
    expose_stage1_transport_operand: str = ""
    # One-variable stage-1 tracer-transport operand arm.  The gate supplies
    # NEMO's native metric zFu/zFv/zFw triplet; momentum and later stages keep
    # the production transport.  Private diagnostic only.
    stage1_tracer_transport_override: object = None
    # One-variable stage-2 tracer-transport operand arm.  The gate supplies
    # NEMO's native metric zFu/zFv/zFw triplet only to the tracer helper;
    # momentum and stages 1/3 keep the production transport.  Private
    # diagnostic only.
    stage2_tracer_transport_override: object = None
    # One-variable stage-3 transport operand injection.  The gate supplies the
    # oracle's native metric zFu/zFv/zFw triplet; conversion back to the shared
    # internal transport representation happens once at the construction
    # boundary.  Private diagnostic only.
    stage3_transport_override: object = None
    # WRITE-only source-order exposure for one WS-RK3 momentum stage.  Empty
    # leaves the returned state untouched; ``hpg``, ``vorticity``, or
    # ``advection`` writes that already-computed production component into the
    # returned u/v slots after the full step.  ``keg`` and ``zad`` name the two
    # compiled halves of the vector-invariant ``advection`` bucket -- NEMO's
    # ``dyn_keg`` and ``dyn_zad``, which ``dyn_adv`` calls in that order under
    # ``ln_dynadv_vec`` (``dynadv.f90:171,176``) -- and are the same already
    # computed production components, published under their own names.  This
    # is a diagnostic seam, not a constructible scheme selector.
    expose_momentum_operator: str = ""
    # Which WS-RK3 stage the exposure above reads.  Stage 2 is where the
    # source-order walk started and stays the default; stage 3 reads the
    # PRODUCTION stage-3 call.  The construction guard above already refuses
    # this hook together with ``expose_stage3_momentum_rhs`` -- they share the
    # returned u/v slots -- so the stage-3 buckets and the stage-3 total are
    # necessarily two runs, and a gate that adds them must say so.
    expose_momentum_operator_stage: int = 2
    # One-variable ablation of NEMO's stage update association.  Public
    # WS-RK3 integrates the full Krhs and only then applies the reference-depth
    # barotropic replacement (stprk3_stg.F90:396-446).  This hook restores the
    # older algebraically equivalent but bit-different pre-projection of Krhs.
    legacy_preproject_stage_rhs: bool = False
    # One-variable ablation of the stage momentum-advection face thickness:
    # restore ``tendencies()``'s own min-of-stretched-T-thicknesses ``h_u``
    # in the flux-form momentum advection at every WS-RK3 site (the
    # step-entry Kbb call, the S-21 stage-1 helper calls, the stage-2/3
    # ``_mom_pert_ws`` calls) instead of NEMO's ``e3u(Kmm) = e3u_0*(1+r3u)``
    # (domzgr_substitute.h90:127; dynadv_up3.F90:160,205-207).  NEMO has no
    # such switch -- e3u is a macro -- so public cards never select this.
    legacy_hadv_min_face_thickness: bool = False
    # One-variable ablation of the stage face-mask rank: restore the 2-D
    # ``state.u_mask`` broadcast over every level in the WS-RK3 stage
    # velocity update and in the stage velocities the tracer transports
    # consume, instead of NEMO's 3-D ``umask(ji,jj,jk)``
    # (``stprk3_stg.F90:367,375,382`` on every stage-1/2 update, ``:444`` on the
    # barotropic correction and ``:273`` on the same correction inside the
    # advective transport).  With the 2-D rule a face that is wet at ANY
    # level keeps the depth-mean increment at EVERY level BELOW its own
    # seabed, and ``dynadv_up3``'s k-slab stencil reads that value where
    # NEMO reads an exact zero.  NEMO has no such switch -- ``umask`` is the
    # 3-D array -- so public cards never select this.
    legacy_2d_stage_face_mask: bool = False
    # WRITE-only stage-3 boundary seam: return the ordinary state immediately
    # before dyn_zdf/tra_zdf in the prognostic slots after the full step has
    # completed.  No public card can construct this diagnostic.
    expose_pre_implicit_state: bool = False
    # WRITE-only companion exposing the actual thickness-form T/S content RHS
    # passed to ``tra_zdf``.  The ordinary solve still completes before these
    # arrays are substituted into the returned diagnostic state.
    expose_pre_implicit_content: bool = False
    # WRITE-only stage-3 tracer boundary before SBC/QSR/LDF content is added.
    # The ordinary step completes before this diagnostic substitutes T/S.
    expose_stage3_advection_content: bool = False
    # One-variable causal arm paired with the seam above.  Replace only the
    # T/S arrays entering the ordinary implicit ZDF solve; production leaves
    # this None.  The gate supplies a source-reconstructed NEMO pre-ZDF seed.
    pre_implicit_tracer_override: object = None
    # The thickness-form path carries the real ``tra_zdf`` content RHS
    # separately from that diagnostic concentration.  This private companion
    # replaces the actual operand for a one-variable oracle arm; no public
    # card can construct it.
    pre_implicit_tracer_content_override: object = None
    # Two-variable FCT/LDF pair discriminator. Replace the helper's
    # already-materialized stage-3 complete-advection content before its
    # ordinary source association; no public card can construct this hook.
    stage3_advection_content_override: object = None
    # Route the already-computed GM/Redi rate into the same stage-3 source
    # in production: NEMO calls tra_ldf before tra_zdf (lines 950-965).
    # Round 110 removed the private selector and introduced no public config;
    # route_gm_redi_stage3_source: bool = False is explicitly retired.
    # One-variable ablation of the stage-3 ZDF thickness time level; restores
    # whole-step-entry rather than faithful N+1/2/Kmm for the discriminator.
    legacy_zdf_entry_kmm_eta: bool = False
    tke_rhs_materialization: str = ""  # Private compiled-RHS boundary walk.
    tke_rhs_intermediate: str = ""  # Private one-output compiled-RHS walk.
    bn2_intermediate: str = ""  # Private one-output compiled-bn2 walk.
    bn2_alpha_beta_override: object = None  # Recorded-entry operator input.
    bn2_tracer_override: object = None  # Recorded-entry T/S operator input.
    tracer_process_trace: object = None; tracer_ldf_diagnostics: object = None
    # Round 189 one-variable production-JIT discriminator.  Replace only the
    # live stage-3 stretch handed to qsr_2BD; the stage geometry, QCO weights,
    # preceding accumulator and returned trajectory retain their own values.
    # This requires the WRITE-only process trace and is not configurable.
    stage3_qsr_stretch_override: object = None
    # Return the stage-3 FCT active-cell map alongside the unchanged process
    # boundaries. Kept separate so Round 136 can prove this larger return
    # graph does not move the already-admitted Round-124 observer's rows.
    tracer_process_branch_activity: bool = False
    # Add the exact tracer-ZDF operands to the process return.  Kept separate
    # so Round 124's smaller return graph remains an unchanged fusion control;
    # Round 126 runs both observers from the same ordinary entry state.
    vertical_solve_trace: bool = False


def rk3_stage_velocity_update(
    velocity_before,
    rhs,
    dt_stage,
    face_mask,
    *,
    vector_form: bool,
    qco_before=None,
    qco_now=None,
    qco_after=None,
):
    """NEMO's RK3 momentum stage time step, both arms, one implementation.

    NEMO selects the arm with the SAME predicate at every stage::

        IF( ln_dynadv_vec .OR. lk_linssh ) THEN   ! applied on velocity
           uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb) + rDt * uu(ji,jj,jk,Krhs) ) * umask(ji,jj,jk)
        ELSE                                      ! thickness weighted
           uu(ji,jj,jk,Kaa) = ( ( 1 + r3u(ji,jj,Kbb) ) * uu(ji,jj,jk,Kbb )
              &               + rDt * ( 1 + r3u(ji,jj,Kmm) ) * uu(ji,jj,jk,Krhs) )
              &             / ( 1 + r3u(ji,jj,Kaa) ) * umask(ji,jj,jk)

    at ``stprk3_stg.F90:365-388`` for stages 1 and 2, and -- this is the part
    legoESM used to miss -- at ``dynzdf.F90:119-142`` for stage 3, because the
    stage-3 time step lives inside ``dyn_zdf``: ``stprk3_stg.F90:395`` CASE(3)
    only adds the leftover RHS terms and does no time stepping at all.

    Module-level rather than a closure so a fidelity gate can drive it with
    NEMO's OWN record arrays and score the result against NEMO's own output; a
    closure cannot be handed a record.  Same reason as
    ``rk3_stage_barotropic_correction``.

    Parameters
    ----------
    velocity_before
        NEMO ``uu(:,:,:,Kbb)``, the BEFORE level, unchanged across the stages.
    rhs
        NEMO ``uu(:,:,:,Krhs)`` for this stage.
    dt_stage
        NEMO ``rDt`` for this stage: ``dt/3``, ``dt/2``, ``dt``.
    face_mask
        NEMO ``umask(ji,jj,jk)``, used by the VECTOR arm only.  The qco arm
        takes no mask here on purpose -- legoESM masks after the barotropic
        correction instead, which is bit-identical because that correction's
        own ``stage_mask`` zeroes a dry face and its ``e3u_0``-weighted column
        mean gives a dry level no weight either way.
    qco_before, qco_now, qco_after
        ``1 + r3u`` at Kbb, Kmm and Kaa.  Required by the qco arm; a missing
        one raises rather than silently degrading to the vector arm.

    Sign/geometry convention: a time step on a horizontal velocity component.
    No vertical axis direction and no flux sign enters it; the thickness
    ratios are positive and dimensionless.
    """
    if not vector_form and (qco_before is None or qco_now is None
                            or qco_after is None):
        raise ValueError(
            "the thickness-weighted arm needs all three (1 + r3u) ratios; got "
            f"before={qco_before is None}, now={qco_now is None}, "
            f"after={qco_after is None} missing")
    if rhs.shape != velocity_before.shape:
        raise ValueError(
            f"rhs {rhs.shape} must match velocity_before "
            f"{velocity_before.shape}")
    if vector_form:
        # dynzdf.F90:121-122 and stprk3_stg.F90:367-368, in NEMO's own
        # association order: add, then mask.
        if face_mask.shape not in (velocity_before.shape,
                                   velocity_before.shape[:-1] + (1,)):
            raise ValueError(
                f"face_mask {face_mask.shape} must match velocity_before "
                f"{velocity_before.shape} or be its level-broadcast form")
        return (velocity_before + dt_stage * rhs) * face_mask
    # dynzdf.F90:127-132 and stprk3_stg.F90:373-378, key_qco form.
    return (qco_before * velocity_before
            + dt_stage * qco_now * rhs) / qco_after


def _nemo_ws_qco_stage_faces(
    eta, h_ref, u_mask_3d, v_mask_3d, grid, *, include_reciprocals=False,
):
    """NEMO ``e3u/e3v(Kmm)`` and ``1 + r3u/r3v`` for one WS-RK3 stage ssh.

    ``dom_qco_r3c_RK3`` (domqco.F90:219-222) builds the U-face free-surface
    ratio as an ``e1e2t``-weighted mean of **ssh** divided by ``hu_0`` --
    NOT the mean of the two ``r3t``, which each divide by their own column's
    ``ht_0`` -- and ``domzgr_substitute.h90:127`` then gives
    ``e3u(Kmm) = e3u_0*(1 + r3u(Kmm)*umask)``.  Taking the ``min`` of the two
    stretched T-cell thicknesses instead is first-order wrong in the ssh
    difference across the face.

    This function assembles OPERANDS only.  The rule -- and the map from
    NEMO's native east/north faces onto legoESM's redundant west/south
    layout -- lives in the single shared
    ``nemo_qco_live_face_geometry_cgrid`` (vertical.py), which the MLF
    tracer transport calls too, because NEMO's own ``dom_qco_r3c``
    (``domqco.F90:166-169``, MLF) and ``dom_qco_r3c_RK3`` (``:219-222``)
    are the same statement.  The operands differ between the two lanes only
    in their SOURCE: the DINO/ORCA cards carry NEMO's own ``hu_0`` and
    ``e1e2*`` on ``z_coord.nemo_*``, while the L1 testcase cards do not, so
    they are built here from the card's own grid and reference ladder
    (``domain.F90:145`` for ``hu_0``).

    The operand construction itself lives in ``vertical.py``'s shared
    ``nemo_qco_card_mesh_operands`` (``e3u_0`` = the min-rule face of the
    REFERENCE thicknesses, ``hu_0`` = ``domain.F90:145``), which the PE
    lane's ``nemo_qco_wzv_operands`` also calls, so a card without NEMO's
    ``mesh_mask.nc`` reaches the NEMO arm on both lanes.

    Inputs and outputs use legoESM's redundant west/south face layout; the
    helper's native east/north extent is mapped once here.

    Returns ``(h_u, h_v, one_plus_r3u, one_plus_r3v)`` with the thicknesses
    3-D and the ratios 2-D.
    """
    dtype = jnp.asarray(h_ref).dtype
    # domain.F90:145 (hu_0 = SUM(e3u_0*umask)) and the C-grid metrics: ONE
    # builder, shared with the PE lane's wzv arm, so the two lanes cannot
    # drift apart in how they reconstruct NEMO's mesh from a card's grid.
    ops = nemo_qco_card_mesh_operands(h_ref, u_mask_3d, v_mask_3d, grid, dtype)
    return nemo_qco_live_face_geometry_cgrid(
        jnp.asarray(eta, dtype=dtype), ops.e3u_0, ops.e3v_0, ops.umask3,
        ops.vmask3, ops.hu_0, ops.hv_0, ops.area_t, ops.area_u, ops.area_v,
        include_reciprocals=include_reciprocals,
    )



def _nemo_dynzdf_drag_face_thickness(
    eta_after, h_bathy, z_coord, config, grid, dtype,
):
    """``e3u_3d(iku)*(1+r3u(Kaa)*umask)`` -- dyn_zdf's bottom-drag divisor.

    The scale factor NEMO's semi-implicit bottom friction divides by
    (``dynzdf.f90:306``, the V twin ``:473``, and the ``ln_dynspg_ts``
    bottom-stress re-add ``:166``/``:168``).  ``e3u_3d`` is the REFERENCE
    three-dimensional face thickness, read from the mesh variable ``e3u_0``
    (``domzgr.f90:186``, ``:201``): over z partial steps the MINIMUM of the
    two neighbouring reference T thicknesses, never their average.

    A named seam, not a second rule: the arithmetic is the one shared
    ``_nemo_ws_qco_stage_faces`` assembler, which the WS-RK3 stage geometry
    and the PE lane's wzv arm also call.  Having it under its own name is
    what lets a test plant the superseded two-cell-average rule into the
    drag path ALONE and show the seamount's velocity move.
    """
    # No non-partial-cell branch: the drag rate this divisor pairs with,
    # ``nemo_bottom_drag_rate_faces``, raises on anything but an
    # ``OceanPartialCellCoordinate`` (ocean_pe_latlon_cgrid.py:4094-4097) and
    # is evaluated FIRST, so a card that got here has one.  GYRE's flat box is
    # a ``masked_zco`` partial-cell coordinate whose cells are all full, which
    # is why the min rule and the average coincide there by construction.
    h_ref = compute_layer_thickness(
        jnp.zeros_like(eta_after), h_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    ).astype(dtype)
    um3, vm3 = compute_face_masks_3d(z_coord.is_active, grid)
    e3u, e3v, _, _ = _nemo_ws_qco_stage_faces(
        eta_after, h_ref, um3.astype(dtype), vm3.astype(dtype), grid)
    return e3u.astype(dtype), e3v.astype(dtype)

def _nemo_metric_stage_transport(metric, face_thickness, corrected_velocity):
    """NEMO ``metric*e3*(velocity+barotropic correction)`` association."""
    return nemo_source_round(
        nemo_source_round(metric[..., None] * face_thickness)
        * corrected_velocity)


def _nemo_cen2_tracer_rhs(
    tracer, p_u, p_v, p_w, e3t_kmm, tmask_kmm, grid,
):
    """Literal RK3 stage-1/2 CEN2 tracer accumulator.

    The FCT card dispatches stages 1-2 to ``traadv_cen.F90:137-149,191-216``
    (``traadv.F90:280-283,355-365``).  Preserve its written face products,
    parenthesised U/V differences, precomputed ``r1_e1e2t``, live
    ``e3t(Kmm)`` division, and the explicit ``wmask(k+1)`` factor on the
    vertical flux (``traadv_cen.F90:201-210``;
    ``dommsk.F90:176,180``).  Stage 3 continues through the single FCT
    implementation and does not call this helper.
    """
    sr = nemo_source_round
    tracer = jnp.asarray(tracer)
    tr_u_sum = centered_cell_to_uface(tracer, nemo_source_sum=True)
    tr_v_sum = interp_to_v_points(tracer, grid, nemo_source_sum=True)
    flux_u = sr(sr(0.5 * p_u) * tr_u_sum)
    flux_v = sr(sr(0.5 * p_v) * tr_v_sum)
    delta_u = sr(flux_u[:, 1:, :] - flux_u[:, :-1, :])
    delta_v = sr(flux_v[1:, :, :] - flux_v[:-1, :, :])
    horizontal_sum = sr(delta_u + delta_v)
    r1_e1e2t = sr(1.0 / jnp.asarray(grid.area_T)[..., None])
    horizontal_scaled = sr(horizontal_sum * r1_e1e2t)
    rhs = sr(-sr(horizontal_scaled / e3t_kmm))

    tracer_w_sum = sr(tracer[..., :-1] + tracer[..., 1:])
    tmask_kmm = jnp.broadcast_to(jnp.asarray(tmask_kmm), tracer.shape)
    wmask_inner = sr(tmask_kmm[..., 1:] * tmask_kmm[..., :-1])
    flux_w_inner = sr(
        sr(sr(0.5 * p_w[..., 1:-1]) * tracer_w_sum) * wmask_inner)
    flux_w = jnp.pad(
        flux_w_inner,
        ((0, 0),) * (flux_w_inner.ndim - 1) + ((1, 1),),
    )
    vertical_difference = sr(flux_w[..., :-1] - flux_w[..., 1:])
    vertical_scaled = sr(vertical_difference * r1_e1e2t)
    return sr(rhs - sr(vertical_scaled / e3t_kmm))


def _nemo_stage_corrected_velocity(
    velocity, transport_average, inverse_depth, barotropic_velocity, face_mask,
    *, return_correction=False,
):
    """Literal ``stprk3_stg.F90:265-278`` zub/zvb composition.

    NEMO uses its separately stored ``uu_b/vv_b(Kmm)`` operand here; reducing
    the 3-D velocity again is algebraically equivalent but not bitwise so.

    ``return_correction`` additionally hands back the written ``zub``/``zvb``
    correction itself.  It is WRITE-ONLY provenance for the fidelity walks,
    which otherwise have to recompute the correction outside this function and
    would then be measuring their own transcription; the returned corrected
    velocity is the same object either way and no production caller sets it.
    """
    correction = nemo_source_round(
        nemo_source_round(transport_average * inverse_depth)
        - barotropic_velocity)
    corrected = nemo_source_round(
        velocity + nemo_source_round(correction[..., None] * face_mask))
    if return_correction:
        return corrected, correction
    return corrected


def nemo_stage_momentum_wzv_resolved(config) -> bool:
    """Does this card's configuration SELECT NEMO's two-solve stage program?

    ``stprk3_stg.f90:356`` takes the vector-invariant arm, ``:358`` skips the
    solve at stage 1, and ``:360`` solves continuity on the RAW stage velocity
    for the momentum vertical advection, while ``traadv.f90:274`` re-solves it
    on the barotropically corrected transports for the tracers.  A card is in
    that program only when it takes the RK3-WS stage program, the
    vector-invariant momentum advection and the literal continuity solve
    together.

    This is the BLAST RADIUS of the second solve, not its execution: a card
    this returns True for MUST set its own explicit
    ``nemo_stage_momentum_wzv_split`` choice (round 163, Decision 55/note
    AT) -- GYRE-zco's is ``True``; Decision 58 makes ORCA2-zps's ``True`` as
    well.  Both are explicit, neither inferred.  The execution predicate reads
    that card choice (or a test hook).
    """
    return (
        getattr(config, "momentum_time_integrator", "euler") == "rk3_ws"
        and getattr(config, "momentum_advection", "flux_form")
        == "vector_invariant"
        and getattr(config, "wzv_call2_evaluation", "generic")
        == "nemo_literal")


def nemo_stage_momentum_wzv_executes(config, hooks=None) -> bool:
    """Does this run ACTUALLY take the second per-stage continuity solve?

    The card has to select the two-solve program.  An EXPLICIT
    ``nemo_stage_momentum_wzv_split`` HOOK (``True`` or ``False``) always
    wins, subject only to ``nemo_stage_momentum_wzv_resolved`` -- that is
    the private one-variable arm rounds 159-162 used to test the candidate
    regardless of any card's production choice, unchanged.

    With no hook, the choice is the CARD's own, read off
    ``config.nemo_stage_momentum_wzv_split`` (round 163, Decision 55/note
    AT) -- an EXPLICIT per-card config field, never inferred from EOS or any
    other unrelated selector (round-163 review BLOCKER: keying it on
    ``eos`` would have been a hidden coupling between unrelated choices).
    GYRE-zco's own resolved config sets it ``True`` (measured, landed), and
    Decision 58 sets ORCA2-zps's own resolved choice to ``True`` after its
    ten-step ladder measurement.  Both choices remain explicit and separate.
    A card that resolves the two-solve program
    WITHOUT setting this field explicitly (``None``, the field's own
    construction default) is a configuration gap, not a silent choice: this
    raises rather than guessing, so the next such card either sets the
    field or is caught before it ships.  This predicate IS the model's
    condition: the Decision-43 card census imports it rather than restating
    it (operator note AR, finding 2).
    """
    if not nemo_stage_momentum_wzv_resolved(config):
        return False
    split = getattr(hooks, "nemo_stage_momentum_wzv_split", None)
    if split is not None:
        return bool(split)
    config_split = getattr(config, "nemo_stage_momentum_wzv_split", None)
    if config_split is None:
        raise ValueError(
            "card resolves NEMO's second per-stage continuity solve "
            "(rk3_ws + vector_invariant + wzv_call2_evaluation=nemo_literal) "
            "but its config does not set nemo_stage_momentum_wzv_split "
            "explicitly -- True or False, never inferred")
    return bool(config_split)


def _nemo_ws_stage_transport(
    stage_velocity, h_stage, stage_index, *, eta_stage, h_ref, Hu_avg, Hv_avg,
    u_mask_3d, v_mask_3d, grid, z_coord, H_bathy, config, dt,
    legacy_min_face_thickness=False, eta_before=None, eta_after=None,
    literal_wzv=False, barotropic_velocity=None, velocity_form_wzv=False,
    momentum_velocity_form_w=False, momentum_wzv_clock=None,
    legacy_wzv_rederived_transport=False,
    legacy_aimp_midpoint_w_metric=False, runoff_mass_flux=None,
):
    """NEMO ``stprk3_stg.F90:257-304`` Kmm stage transport triplet.

    ``e3u(Kmm)`` is NEMO's qco face thickness ``e3u_0*(1 + r3u(Kmm))``
    (``domqco.F90:219-222``, ``domzgr_substitute.h90:127``), built by
    ``_nemo_ws_qco_stage_faces`` from ``eta_stage``.

    ``zFu = e2u*e3u(Kmm)*(uu(Kmm) + zub)`` with ``zub = un_adv/hu(Kmm) -
    uu_b(Kmm)`` (:270-277): the stage velocity's thickness mean is replaced
    by the external time-mean transport, and ``ww`` follows from continuity of
    that transport with the qco stage-thickness term (``wzv`` np_transport,
    sshwzv.F90:315-335).  Stage 3 splits ``ww`` into the explicit advection
    share and the implicit ``wi`` (``wAimp``, :298-302); stages 1-2 carry
    ``wi = 0`` (:287).  ONE triplet per stage feeds both ``dyn_adv``
    (:315,331-334) and ``tra_adv`` (:456-519).  Returns
    ``(mf_u, mf_v, w_explicit, h_stage, hu_stage, hv_stage, wi_stage, zFu,
    zFv, corrected_u, corrected_v, w_momentum)``.  ``w_explicit`` is the
    TRACER transport's field (``traadv.f90:274``, the transport indicator) and
    ``w_momentum`` is the momentum vertical advection's own solve on the raw
    stage velocity (``stprk3_stg.f90:360``, the velocity indicator), or
    ``None`` on cards that do not run that second solve.  The native pair retains NEMO's product association;
    dividing it back to a metric-free flux and multiplying again is not
    bitwise equivalent on the rotated GYRE grid.
    """
    u_stage, v_stage = stage_velocity
    h_stage = nemo_qco_live_t_thickness(
        eta_stage, H_bathy, z_coord, jnp.asarray(h_stage).dtype,
        e3t_0=h_ref)
    if legacy_min_face_thickness:
        # Private ablation control ONLY (_NEMOWSRK3TestHooks); NEMO has no
        # such switch and the WS-RK3 identity never selects this arm.
        hu_stage = min_cell_to_uface(h_stage)
        hv_stage = min_cell_to_vface(h_stage, grid)
        r1_hu_stage = r1_hv_stage = None
    else:
        (hu_stage, hv_stage, _, _, r1_hu_stage,
         r1_hv_stage) = _nemo_ws_qco_stage_faces(
            eta_stage, h_ref, u_mask_3d, v_mask_3d, grid,
            include_reciprocals=True)
    Hu_stage_depth = jnp.sum(hu_stage, axis=-1)
    Hv_stage_depth = jnp.sum(hv_stage, axis=-1)
    if barotropic_velocity is None:
        Hu_stage_raw = jnp.sum(hu_stage * u_stage, axis=-1)
        Hv_stage_raw = jnp.sum(hv_stage * v_stage, axis=-1)
        u_stage_corr = u_stage + (
            (Hu_avg - Hu_stage_raw)
            / jnp.maximum(Hu_stage_depth, 1.0e-10))[..., jnp.newaxis]
        v_stage_corr = v_stage + (
            (Hv_avg - Hv_stage_raw)
            / jnp.maximum(Hv_stage_depth, 1.0e-10))[..., jnp.newaxis]
    else:
        # stprk3_stg.F90:265-278 keeps uu_b/vv_b(Kmm) as separate
        # prognostics.  Preserve both the reciprocal and subtract boundaries;
        # re-reducing the 3-D field is algebraically equal after :433-446 but
        # changes the last bits on GYRE's stage-3 transport.
        r1_hu = (r1_hu_stage if r1_hu_stage is not None else
                 nemo_source_round(
                     1.0 / jnp.maximum(Hu_stage_depth, 1.0e-10)))
        r1_hv = (r1_hv_stage if r1_hv_stage is not None else
                 nemo_source_round(
                     1.0 / jnp.maximum(Hv_stage_depth, 1.0e-10)))
        u_stage_corr = _nemo_stage_corrected_velocity(
            u_stage, Hu_avg, r1_hu, barotropic_velocity[0], u_mask_3d)
        v_stage_corr = _nemo_stage_corrected_velocity(
            v_stage, Hv_avg, r1_hv, barotropic_velocity[1], v_mask_3d)
    mf_u = hu_stage * u_stage_corr * u_mask_3d
    mf_v = hv_stage * v_stage_corr * v_mask_3d
    zfu_stage = _nemo_metric_stage_transport(
        jnp.asarray(grid.dy_u), hu_stage, u_stage_corr)
    zfv_stage = _nemo_metric_stage_transport(
        jnp.asarray(grid.dx_v), hv_stage, v_stage_corr)
    stage_div = divergence_cgrid(mf_u, mf_v, grid)
    w_momentum = None
    if literal_wzv:
        # sshwzv.F90:331-336 (qco arm), entered from stprk3_stg.F90:297 as
        # ``wzv(kstp, Kbb, Kmm, Kaa, zFu, zFv, ww, np_transport)``: the SAME
        # routine the MLF lane calls, so legoESM calls the same function --
        # ``e2u*e3u(Kmm)*(uu+zub)`` differenced and divided by ``e1e2t`` and
        # live ``e3t(Kmm)`` (divhor.F90:116-123,139-141), then the
        # ``r1_Dt*e3t_0*(r3t(Kaa)-r3t(Kbb))`` stretching term.
        _tmask3 = (
            z_coord.is_active.astype(h_stage.dtype)
            if isinstance(z_coord, OceanPartialCellCoordinate)
            else jnp.ones_like(h_stage))
        w_stage, _, _ = nemo_qco_wzv_operands(
            eta_stage, eta_before, u_stage, v_stage, grid, z_coord,
            u_mask_3d, v_mask_3d, _tmask3, dt,
            eta_after_override=eta_after,
            transport_after_override=(
                (Hu_avg, Hv_avg) if legacy_wzv_rederived_transport else None),
            barotropic_velocity_override=(
                barotropic_velocity if legacy_wzv_rederived_transport else None),
            volume_transport_override=(
                # divhor.f90:123-130 (velocity indicator) rebuilds each face
                # transport from the raw stage velocity; :132-138 (transport
                # indicator) differences the already corrected zFu/zFv.
                None if (legacy_wzv_rederived_transport or velocity_form_wzv)
                else (zfu_stage, zfv_stage)),
            runoff_mass_flux=runoff_mass_flux)
        if momentum_velocity_form_w:
            # THE SECOND SOLVE.  stprk3_stg.f90:360 hands wzv the raw stage
            # velocity under the velocity indicator and the momentum vertical
            # advection reads the result; traadv.f90:274 then re-solves
            # continuity on the corrected transports and overwrites the same
            # array before the tracer's vertical transport is formed at :280.
            # ``w_stage`` above is that second, tracer-owned field and is left
            # byte for byte as it was; this is the first one.
            # The private clock arm replaces NEMO's (rDt, r3t(Kaa)) PAIR at
            # once; production passes the step clock with the barotropic
            # after level, which is the same product algebraically.
            _w_dt, _w_eta_after = (
                (dt, eta_after) if momentum_wzv_clock is None
                else momentum_wzv_clock)
            w_momentum, _, _ = nemo_qco_wzv_operands(
                eta_stage, eta_before, u_stage, v_stage, grid, z_coord,
                u_mask_3d, v_mask_3d, _tmask3, _w_dt,
                eta_after_override=_w_eta_after,
                volume_transport_override=None,
                runoff_mass_flux=runoff_mass_flux)
    else:
        w_stage = diagnose_w_from_flux_div(
            stage_div, z_coord, thickness_weighted=True)
    wi_stage = jnp.zeros_like(w_stage)
    if (stage_index == 2
            and getattr(config, "adaptive_implicit_vertadv", False)):
        if w_momentum is not None:
            # stprk3_stg.f90:362 partitions the MOMENTUM pair under the
            # velocity indicator and traadv.f90:277 partitions the TRACER pair
            # under the transport indicator: two partitions, not one.  Only
            # one is transcribed, so refuse loudly rather than hand one
            # partition to two consumers.
            raise ValueError(
                "NEMO's second continuity solve and adaptive-implicit "
                "vertical advection need TWO stage-3 partitions "
                "(stprk3_stg.f90:362 and traadv.f90:277); only the transport "
                "one is transcribed, so this pair is refused")
        if not legacy_aimp_midpoint_w_metric:
            from legoesm.ocean.physics.vertical_mixing import nemo_e3w_kmm

            # domzgr_substitute.h90:131: e3w(Kmm)=e3w_0*(1+r3t(Kmm));
            # domqco.F90:209: r3t=ssh/ht_0.  The canonical helper returns
            # NEMO jk=2..jpk interior interfaces; boundary entries are unused
            # by wAimp_RK3_t and are padded only to the local nlev+1 layout.
            stretch = 1.0 + eta_stage / jnp.maximum(
                jnp.sum(h_ref, axis=-1), 1.0e-10)
            e3w_int = nemo_e3w_kmm(z_coord, h_stage, stretch)
            e3w = jnp.concatenate(
                [e3w_int[..., :1], e3w_int, e3w_int[..., -1:]], axis=-1)
        else:
            e3w_int = 0.5 * (h_stage[..., :-1] + h_stage[..., 1:])
            e3w = jnp.concatenate(
                [h_stage[..., :1], e3w_int, h_stage[..., -1:]], axis=-1)
        split = nemo_wicker_aimp_partition_transport(
            mf_u, mf_v, w_stage, h_stage, e3w,
            grid.area_T, grid.dy_u, grid.dx_v, dt)
        w_stage, wi_stage = split.w_explicit, split.w_implicit
    return (
        mf_u, mf_v, w_stage, h_stage, hu_stage, hv_stage, wi_stage,
        zfu_stage, zfv_stage, u_stage_corr, v_stage_corr, w_momentum)


class _NEMOWSBarotropicTrace(NamedTuple):
    """Private WRITE-only equivalent returned by the GYRE boundary gate."""

    state_after_barotropic: object
    substeps: object
    slow_forcing: object
    slow_forcing_operands: object
    transport_average: object


class _NEMOWSLiveOperandTrace(NamedTuple):
    """Private WRITE-only operands returned by the round-51 live-path gate."""

    state_after: object
    stage_states: object
    operator_operands: object
    stage_geometry: object
    stage_qco: object
    stage_coefficients: object
    tke_entry: object
    tke_statement_trace: object
    barotropic_targets: object
    slow_forcing_producer: object
    stage_rhs: object
    stage1_full_rhs: object
    stage1_rhs_walk: object
    stage_raw_velocities: object
    barotropic_correction_geometry: object
    stage_outputs: object
    stage_tracer_sources: object


def _nemo_qsr_stage3_rate(
    tendency_kbb, qsr_kbb, qsr_kmm, thickness_kbb, thickness_kmm,
):
    """Replace the Kbb qsr component with NEMO's live Kmm stage-3 qsr."""
    return (
        (tendency_kbb - qsr_kbb) * thickness_kbb
        / jnp.maximum(thickness_kmm, 1.0e-10)
        + qsr_kmm
    )


def _nemo_ws_stage_barotropic_velocity(
    stage: int,
    kbb_velocity: tuple,
    nnn_velocity: tuple,
) -> tuple:
    """Select NEMO's separately carried Kmm pair for one RK3 stage.

    ``stprk3.F90:186`` computes ``uu_b/vv_b(Naa)`` once before all stages.
    Stage 1 is called with ``Kmm=Nbb`` at :195, then :197 swaps that freshly
    solved slot into ``Nnn``; stages 2 and 3 therefore read this-step ``Nnn``
    at :200-207.  Only stage 1 reads the last-step ``Kbb`` pair.  Keeping this
    choice in a tiny fail-closed helper makes the time-level contract directly
    testable and prevents a same-step external target from leaking into S-21.
    """
    if stage == 1:
        return kbb_velocity
    if stage in (2, 3):
        return nnn_velocity
    raise ValueError(f"NEMO WS-RK3 stage must be 1, 2, or 3; got {stage}")


def _nemo_ws_rk3_tracer_pair_step(
    tr_a: jnp.ndarray,
    tr_b: jnp.ndarray,
    tracer_advection: str,
    mass_flux_u: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    w_baro: jnp.ndarray,
    h_k_old: jnp.ndarray,
    h_k_new: jnp.ndarray,
    h_u_old: jnp.ndarray,
    h_v_old: jnp.ndarray,
    grid,
    dt: float,
    active_3d: jnp.ndarray,
    recon_fill_mask: jnp.ndarray | None = None,
    linssh_top_flux: bool = False,
    stage_transport_geometry=None,
    fct_low_order_predictor: str = "nemo_rk3_two_step",
    bbl_context=None,
    stage_source_rates=None,
    stage_source_terms=None,
    stage_qco_weights=None,
    stop_after_stage: int = 3,
    resume=None,
    stage3_advection_content_override=None,
    return_final_content: bool = False,
    return_stage1_trace: bool = False,
    return_fct_activity: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """NEMO key_RK3 tracer stage program (Wicker--Skamarock form).

    ``resume=(k, a_k, b_k)`` hands in the already-advanced stage-``k`` pair
    (k = 1 or 2) so stages ``<= k`` are not recomputed: the momentum program
    builds the stage-1/2 tracers as the stage-2/3 ``eos+dyn_hpg`` operands
    (stprk3_stg.F90:317-320) and the tracer program then finishes stage 3 on
    the SAME arrays (``:456-519``: one stage ladder, not two).

    ``stprk3_stg.F90:112-249,519-559`` restarts every stage from Kbb and
    applies the previous stage RHS with ``dt/3``, ``dt/2``, then ``dt``.
    Under key_qco the stage result is thickness weighted; the one-third and
    one-half thicknesses below are the corresponding linear r3t stages.
    The flux reconstruction receives the live stage interval so FCT uses the
    same stage Courant factor.  This selector changes no existing ``rk3``
    (SSP) user.
    """
    if stage_transport_geometry is None:
        stage_transport_geometry = (
            (mass_flux_u, mass_flux_v, w_baro, h_k_old, h_u_old, h_v_old),
        ) * 3
    if len(stage_transport_geometry) != 3:
        raise ValueError("stage_transport_geometry must contain exactly 3 stages")
    if stop_after_stage not in (1, 2, 3):
        raise ValueError("stop_after_stage must be 1, 2, or 3")
    if stage_source_rates is None:
        zero_a = jnp.zeros_like(tr_a)
        zero_b = jnp.zeros_like(tr_b)
        stage_source_rates = ((zero_a, zero_b),) * 3
    if len(stage_source_rates) != 3:
        raise ValueError("stage_source_rates must contain exactly 3 stages")
    if stage_source_terms is not None and len(stage_source_terms) != 3:
        raise ValueError("stage_source_terms must contain exactly 3 stages")
    if stage_qco_weights is not None and len(stage_qco_weights) != 3:
        raise ValueError("stage_qco_weights must contain exactly 3 stages")
    resume_stage = 0
    if resume is not None:
        resume_stage, a_resume, b_resume = resume
        if resume_stage not in (1, 2) or resume_stage >= stop_after_stage:
            raise ValueError(
                "resume must hand in stage 1 or 2, below stop_after_stage")

    def _flux_pair(a_val, b_val, stage_dt, stage_index, h_after):
        fct_activity = None
        stage_geom = stage_transport_geometry[stage_index]
        if len(stage_geom) == 6:
            mf_u, mf_v, w_stage, h_stage, hu_stage, hv_stage = stage_geom
            wi_stage = None
            zfu_stage = zfv_stage = None
        elif len(stage_geom) == 7:
            (mf_u, mf_v, w_stage, h_stage, hu_stage, hv_stage,
             wi_stage) = stage_geom
            zfu_stage = zfv_stage = None
        elif len(stage_geom) == 9:
            (mf_u, mf_v, w_stage, h_stage, hu_stage, hv_stage,
             wi_stage, zfu_stage, zfv_stage) = stage_geom
        elif len(stage_geom) == 11:
            (mf_u, mf_v, w_stage, h_stage, hu_stage, hv_stage,
             wi_stage, zfu_stage, zfv_stage, _, _) = stage_geom
        elif len(stage_geom) == 12:
            # Slot 11 is the MOMENTUM vertical velocity of NEMO's first
            # per-stage continuity solve (stprk3_stg.f90:360).  The tracer
            # transport re-solves continuity for itself (traadv.f90:274), so
            # this path keeps reading slot 2 and never slot 11.
            (mf_u, mf_v, w_stage, h_stage, hu_stage, hv_stage,
             wi_stage, zfu_stage, zfv_stage, _, _, _) = stage_geom
        else:
            raise ValueError(
                "each stage transport geometry needs 6, 7, 9, 11, or 12 "
                "arrays")
        # NEMO key_RK3 runs the FCT limiter at stage 3 ONLY: traadv.F90:281-282
        # forces ``ll_dofct = .FALSE.`` for ``kstg /= 3`` and :361-364 then
        # dispatches ``np_FCT`` to ``tra_adv_cen(nn_fct_h, nn_fct_v)``, the
        # plain 2nd-order centred flux (traadv_cen.F90:140-149, 196-210) with
        # the stage transport and Kmm tracer.  There is no NEMO switch here,
        # so there is none in legoESM: the certified WS card selects ``fct2``
        # (= nn_fct_h/v = 2) and stages 1-2 take the matching ``centered``
        # flux.  Running the limiter at every stage put the front cells'
        # stage-1 increment at 0x / 2x NEMO's (the upwind low-order
        # signature), which is what fed the stage-2 EOS/HPG operands.
        if tracer_advection == "fct2" and stage_index < 2:
            # Literal CEN2 accumulator used by key_RK3 at stages 1/2.
            # traadv.F90:280-283 disables FCT before stage 3, then
            # traadv_cen.F90:137-149,203-216 forms metric-bearing face
            # fluxes, differences their already-rounded products, multiplies
            # by r1_e1e2t, and finally divides by e3t(Kmm).  The shared generic
            # content-divergence helper algebraically cancels that last
            # divide/multiply and moves GYRE's tracer Kaa by O(1e-11), so keep
            # this source association inside the single NEMO WS identity.
            p_u = (
                zfu_stage if zfu_stage is not None
                else mf_u * jnp.asarray(grid.dy_u)[..., None])
            p_v = (
                zfv_stage if zfv_stage is not None
                else mf_v * jnp.asarray(grid.dx_v)[..., None])
            p_w = w_stage * jnp.asarray(grid.area_T)[..., None]

            def _cen2_content_div(tracer):
                rhs = _nemo_cen2_tracer_rhs(
                    tracer, p_u, p_v, p_w, h_stage, active_3d, grid)
                return nemo_source_round(
                    -nemo_source_round(h_stage * rhs)), rhs

            (fd_a, rhs_a), (fd_b, rhs_b) = (
                _cen2_content_div(a_val), _cen2_content_div(b_val))
        else:
            pair_result = compute_advection_flux_div_pair(
                a_val, b_val, tracer_advection, mf_u, mf_v,
                w_stage, h_stage, hu_stage, hv_stage, grid, stage_dt,
                recon_fill_mask=recon_fill_mask,
                linssh_top_flux=linssh_top_flux,
                tr_a_before=tr_a,
                tr_b_before=tr_b,
                fct_low_order_predictor=fct_low_order_predictor,
                fct_base_thickness=h_k_old,
                fct_after_thickness=h_after,
                fct_implicit_w=wi_stage,
                return_a_fct_activity=(
                    return_fct_activity and stage_index == 2),
            )
            if return_fct_activity and stage_index == 2:
                (dh_a, dv_a), (dh_b, dv_b), fct_activity = pair_result
            else:
                (dh_a, dv_a), (dh_b, dv_b) = pair_result
                fct_activity = None
            fd_a, fd_b = dh_a + dv_a, dh_b + dv_b
            rhs_a = rhs_b = None
        if stage_index == 2 and bbl_context is not None:
            from legoesm.ocean.physics.bbl_adv import (
                apply_bbl_diffusive_tendency,
                apply_bbl_adv_tendency,
                bbl_transports,
                nemo_bbl_diffusive_coefficients,
            )
            (adv_option, diffusive_option, geom, diffusive_geom,
             area, dy_u, dx_v, gamma_s, aht_m2_s, rho0,
             nemo_reference_geometry, bbl_grid, bbl_eos_form) = bbl_context
            if diffusive_option == 1:
                # trabbl.F90:348 reads gdept(bottom,Kmm).  Under key_qco that
                # is gdept_0(bottom)*(1+r3t(Kmm)); the stage thickness sum is
                # H*(1+r3t), so this ratio is the identical stage stretch.
                reference_depth = jnp.sum(h_k_old, axis=-1)
                stage_stretch = jnp.sum(h_stage, axis=-1) / jnp.maximum(
                    reference_depth, jnp.asarray(1.0e-10, h_stage.dtype))
                live_bottom_depth = (
                    diffusive_geom.dep_bot_ref * stage_stretch)
            elif nemo_reference_geometry:
                reference_depth = jnp.sum(geom.h_ref, axis=-1)
                stage_stretch = jnp.sum(h_stage, axis=-1) / jnp.maximum(
                    reference_depth, jnp.asarray(1.0e-10, h_stage.dtype))
                live_bottom_depth = geom.dep_bot * stage_stretch
            else:
                live_depth = jnp.cumsum(h_stage, axis=-1) - 0.5 * h_stage
                live_bottom_depth = jnp.take_along_axis(
                    live_depth, geom.bot_k[..., None], axis=-1)[..., 0]
            zero_a, zero_b = jnp.zeros_like(tr_a), jnp.zeros_like(tr_b)
            bbl_a, bbl_b = zero_a, zero_b
            if adv_option == 2:
                utr, vtr = bbl_transports(
                    tr_a, tr_b, geom, dy_u, dx_v,
                    gamma_s=gamma_s, rho_0=rho0,
                    bottom_depth_m=live_bottom_depth,
                )
                bbl_a, bbl_b = apply_bbl_adv_tendency(
                    bbl_a, bbl_b, tr_a, tr_b, h_stage, area, geom, utr, vtr,
                    nlev=tr_a.shape[-1],
                )
            if diffusive_option == 1:
                ahu_bbl, ahv_bbl = nemo_bbl_diffusive_coefficients(
                    tr_a, tr_b, diffusive_geom,
                    bottom_depth_m=live_bottom_depth, rho_0=rho0,
                    grid=bbl_grid, eos_form=bbl_eos_form,
                )
                bbl_a, bbl_b = apply_bbl_diffusive_tendency(
                    bbl_a, bbl_b, tr_a, tr_b, h_stage, area,
                    diffusive_geom, ahu_bbl, ahv_bbl, grid=bbl_grid,
                )
            # tra_bbl adds a concentration tendency to Krhs using Kbb tracers
            # and Kmm volume (stprk3_stg.F90:588; trabbl.F90:129-136,243-284).
            # This helper evolves content, so convert h*Krhs to a negative
            # flux divergence before the final dt update.
            fd_a = fd_a - h_stage * bbl_a
            fd_b = fd_b - h_stage * bbl_b
        return fd_a, fd_b, rhs_a, rhs_b, fct_activity

    def _source_order_sum(base, fallback, terms):
        if terms is None:
            return base + fallback
        out = base
        for term in terms:
            out = nemo_source_round(out + term)
        return out

    def _stage(
        base, flux_div, concentration_rhs, source_rate, stage_dt,
        h_after, h_rhs, stage_index, source_terms=None,
    ):
        if stage_qco_weights is not None and concentration_rhs is not None:
            q_before, q_rhs, q_after = stage_qco_weights[stage_index]
            rhs = _source_order_sum(
                concentration_rhs, source_rate, source_terms)
            out = (
                q_before[..., None] * base
                + stage_dt * q_rhs[..., None] * rhs
            ) / q_after[..., None]
            return jnp.where(active_3d > 0.5, out, base)
        out = (
            h_k_old * base - stage_dt * flux_div
            + stage_dt * h_rhs * source_rate
        ) / jnp.maximum(h_after, 1.0e-10)
        return jnp.where(active_3d > 0.5, out, base)

    h_one_third = h_k_old + (h_k_new - h_k_old) / 3.0
    h_one_half = 0.5 * (h_k_old + h_k_new)
    if resume_stage >= 1:
        a1, b1 = a_resume, b_resume
    else:
        fd0_a, fd0_b, rhs0_a, rhs0_b, _ = _flux_pair(
            tr_a, tr_b, dt / 3.0, 0, h_one_third)
        # WRITE-only source-order values.  The CEN2 RK stage already returns
        # its concentration RHS directly; for another private diagnostic
        # configuration, undo the helper's content convention exactly once.
        trace_adv_a = (
            rhs0_a if rhs0_a is not None
            else -fd0_a / jnp.maximum(h_k_old, 1.0e-10))
        trace_adv_b = (
            rhs0_b if rhs0_b is not None
            else -fd0_b / jnp.maximum(h_k_old, 1.0e-10))
        _terms0 = None if stage_source_terms is None else stage_source_terms[0]
        trace_sbc_a = _source_order_sum(
            trace_adv_a, stage_source_rates[0][0],
            None if _terms0 is None else _terms0[0])
        trace_sbc_b = _source_order_sum(
            trace_adv_b, stage_source_rates[0][1],
            None if _terms0 is None else _terms0[1])
        a1 = _stage(
            tr_a, fd0_a, rhs0_a, stage_source_rates[0][0], dt / 3.0,
            h_one_third, h_k_old, 0,
            None if _terms0 is None else _terms0[0])
        b1 = _stage(
            tr_b, fd0_b, rhs0_b, stage_source_rates[0][1], dt / 3.0,
            h_one_third, h_k_old, 0,
            None if _terms0 is None else _terms0[1])
    if stop_after_stage == 1:
        if return_stage1_trace:
            return (
                a1, b1, trace_adv_a, trace_adv_b,
                trace_sbc_a, trace_sbc_b,
            )
        return a1, b1
    if resume_stage >= 2:
        a2, b2 = a_resume, b_resume
    else:
        fd1_a, fd1_b, rhs1_a, rhs1_b, _ = _flux_pair(
            a1, b1, dt / 2.0, 1, h_one_half)
        _terms1 = None if stage_source_terms is None else stage_source_terms[1]
        a2 = _stage(
            tr_a, fd1_a, rhs1_a, stage_source_rates[1][0], dt / 2.0,
            h_one_half, h_one_third, 1,
            None if _terms1 is None else _terms1[0])
        b2 = _stage(
            tr_b, fd1_b, rhs1_b, stage_source_rates[1][1], dt / 2.0,
            h_one_half, h_one_third, 1,
            None if _terms1 is None else _terms1[1])
    if stop_after_stage == 2:
        return a2, b2
    fd2_a, fd2_b, _, _, fct_activity = _flux_pair(a2, b2, dt, 2, h_k_new)
    advection_content_a = h_k_old * tr_a - dt * fd2_a
    advection_content_b = h_k_old * tr_b - dt * fd2_b
    if stage3_advection_content_override is not None:
        advection_content_a, advection_content_b = (
            stage3_advection_content_override)
    content_a = (advection_content_a
                 + dt * h_one_half * stage_source_rates[2][0])
    content_b = (advection_content_b
                 + dt * h_one_half * stage_source_rates[2][1])
    out_a = jnp.where(
        active_3d > 0.5,
        content_a / jnp.maximum(h_k_new, 1.0e-10), tr_a)
    out_b = jnp.where(
        active_3d > 0.5,
        content_b / jnp.maximum(h_k_new, 1.0e-10), tr_b)
    if return_final_content and return_fct_activity:
        return (
            out_a, out_b, content_a, content_b,
            advection_content_a, advection_content_b, fct_activity,
        )
    if return_final_content:
        return (
            out_a, out_b, content_a, content_b,
            advection_content_a, advection_content_b,
        )
    return out_a, out_b


def _forward_backward_coriolis_3d(
    u: jnp.ndarray,
    v: jnp.ndarray,
    dt: float,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    land_mask: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply forward-backward (Matsuno) Coriolis to baroclinic perturbation velocity.

    The forward-backward scheme is unconditionally neutral for the
    inertial oscillation: the amplification factor is exactly 1,
    unlike forward Euler which amplifies by sqrt(1 + (f*dt)^2).

    Steps:
      1. Compute perturbation velocity u' = u - U_bar, v' = v - V_bar
      2. Forward:  u'_new = u' + dt * f_u * avg(v'  -> u-points)
      3. Backward: v'_new = v' - dt * f_v * avg(u'_new -> v-points)
      4. Reconstruct: u_new = u'_new + U_bar, v_new = v'_new + V_bar

    Parameters
    ----------
    u : (n_lat, n_lon+1, nlev) full 3D u velocity
    v : (n_lat+1, n_lon, nlev) full 3D v velocity
    dt : time step [s]
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    config : LatLonCGridOceanConfig
    u_mask : (n_lat, n_lon+1)
    v_mask : (n_lat+1, n_lon)
    land_mask : (n_lat, n_lon)
    eta, H_bathy : (n_lat, n_lon) for layer thickness computation

    Returns
    -------
    u_new, v_new : updated velocities with Coriolis applied
    """
    u.shape[-1]

    u_mask_3d = u_mask[..., jnp.newaxis]
    v_mask_3d = v_mask[..., jnp.newaxis]

    # --- Layer thickness at face points for depth averaging ---
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta, eta_floor) * land_mask

    h_k = compute_layer_thickness(
        eta_safe, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )

    # h at u/v-faces — min-rule (MOM6/MITgcm hFacW convention).
    # Must match the PE tendency and slow-forcing depth-average which
    # both use min_cell_to_uface/min_cell_to_vface.  Arithmetic mean
    # overestimates face depth at topographic steps, creating a
    # barotropic-baroclinic residual that drives spurious currents.
    h_u = min_cell_to_uface(h_k)
    h_v = min_cell_to_vface(h_k, grid)

    # --- Depth-averaged velocity (barotropic component) ---
    # Thickness-weighted depth average masked by the face mask (#517
    # item 1: shared depth_average_to_faces; floor = min_water_col,
    # passed verbatim → bit-identical).
    U_bar = depth_average_to_faces(u, h_u, u_mask, min_water_col)
    V_bar = depth_average_to_faces(v, h_v, v_mask, min_water_col)

    # --- Perturbation velocity ---
    u_prime = (u - U_bar[..., jnp.newaxis]) * u_mask_3d
    v_prime = (v - V_bar[..., jnp.newaxis]) * v_mask_3d

    # --- Coriolis parameter at face points (#517: shared fold-safe helper) ---
    from legoesm.ocean.dynamics.barotropic_common import coriolis_at_faces
    f_u, f_v = coriolis_at_faces(grid, u.dtype)  # (n_lat, n_lon+1), (n_lat+1, n_lon)

    # --- Forward step: update u' using old v' ---
    # Average v' to u-points (Sadourny 4-point average)
    v_west = jnp.roll(v_prime, 1, axis=1)
    v_at_u = 0.25 * (v_prime[:-1] + v_prime[1:]
                      + v_west[:-1] + v_west[1:])
    v_at_u = jnp.concatenate([v_at_u, v_at_u[:, 0:1, :]], axis=1)

    u_prime_new = (u_prime + dt * f_u[:, :, jnp.newaxis] * v_at_u) * u_mask_3d

    # --- Backward step: update v' using NEW u' ---
    # Average u'_new to v-points (Sadourny 4-point average) via the
    # shared cell-pad-first helper: the MPI band partition-cut v-face
    # averages the neighbour rank's true u' row instead of the old
    # interior-then-pad_ns_vector_u refill (one face row off at a cut).
    # Wall BC at physical poles; fold (sign*perm) on tripolar — serial
    # bit-identical.
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        interp_u_to_vface_4pt,
    )
    u_at_v = interp_u_to_vface_4pt(u_prime_new, grid)

    v_prime_new = (v_prime - dt * f_v[:, :, jnp.newaxis] * u_at_v) * v_mask_3d

    # --- Reconstruct full velocity ---
    u_new = u_prime_new + U_bar[..., jnp.newaxis]
    v_new = v_prime_new + V_bar[..., jnp.newaxis]

    return u_new, v_new


def _eke_av_at_interior_wfaces(A_v_phys, A_v_bg, nlev, prefix_shape):
    """Vertical viscosity at the ``M-1 = nlev-2`` INTERIOR W-interfaces, for the
    3-D EKE implicit vertical diffusion (the ``eke_3d`` path).

    The 3-D EKE field lives on the ``nlev-1`` interior T-interfaces (the W-grid);
    its implicit vertical diffusion needs the diffusivity at the ``nlev-2``
    interfaces BETWEEN those W-levels, which sit at the interior T-centres
    ``k = 1 .. nlev-2`` (interior W-interface ``k`` is at T-centre ``k+1``).
    ``A_v_phys`` may be:

    - ``None`` — use the constant background ``A_v_bg`` (the ACC case: the TKE
      path leaves ``tend.A_v`` unset);
    - cell-centred (last axis ``nlev``) — the interior T-centres are ``[...,1:nlev-1]``;
    - at the T-interfaces (last axis ``nlev-1``; the shape the vertical-mixing
      modules KPP/TKE/Richardson produce) — average adjacent T-interfaces to the
      interior T-centres (Veros's ``0.5*(kappaM[k]+kappaM[k+1])``).

    Returns ``(*prefix_shape, nlev-2)``, ``>= 0``.  ``A_v_bg`` carries the dtype.
    """
    if A_v_phys is None:
        return jnp.broadcast_to(A_v_bg, tuple(prefix_shape) + (nlev - 2,))
    A_v_p = jnp.asarray(A_v_phys, dtype=A_v_bg.dtype) + A_v_bg
    if A_v_p.shape[-1] == nlev:                 # cell-centred -> interior T-centres
        return A_v_p[..., 1:nlev - 1]
    if A_v_p.shape[-1] == nlev - 1:             # T-interfaces -> avg to interior T-centres
        return 0.5 * (A_v_p[..., :-1] + A_v_p[..., 1:])
    raise ValueError(
        "eke_3d vertical diffusion: A_v_phys last axis must be nlev (cell-centred) "
        f"or nlev-1 (interfaces); got {A_v_p.shape[-1]} (nlev={nlev})."
    )


def static_kappa_redi_override(gm_cfg, grid):
    """Per-column kappa_Redi override for kappa_redi_lat_scaling.

    NEMO ``nn_aht_ijk_t=20`` (``ldftra.F90:325-329`` -> ``ldf_c2d('TRA', ...)``,
    ``ldfc1d_c2d.F90:141-145``) does NOT build one T-point field and average it
    onto the faces: ``ahtu`` and ``ahtv`` are TWO INDEPENDENTLY-EVALUATED
    arrays, ``ahtu(ji,jj) = zUfac·MAX(e1u,e2u)^inn`` and
    ``ahtv(ji,jj) = zUfac·MAX(e1v,e2v)^inn`` — i.e. ``∝ cos(lat_u)`` and
    ``∝ cos(lat_v)`` respectively, each AT ITS OWN POINT.  On this regular
    lat-lon grid ``cos(lat_u) == cos(lat_T)`` exactly (the u-face shares its
    T-row's latitude), so the u-face/T-point field below is exact; the
    v-face sits at a genuinely different (shifted) latitude and needs its own
    ``cos(lat_v)`` evaluation — reusing the T-point field for both (the
    pre-#1226-tier-2 code) left a real ~1e-5-level v-face amplitude error
    that does NOT vanish with grid refinement in the same way (it is not an
    interpolation artifact; there never was an interpolation).

    Returns ``(kappa_T, kappa_v)``: ``kappa_T`` = ``(n_lat, n_lon)``
    ``kappa_Redi·cos(lat_T)`` (feeds the T-point broadcast used by zfu/zft
    unchanged); ``kappa_v`` = ``(n_lat, n_lon)`` ``kappa_Redi·cos(lat_v)`` at
    the v-face (north-face-of-cell-j convention, matching ``grid.dx_v[1:,:]``)
    for the v-face-specific consumer (``kappa_Redi_v`` /
    ``kappa_redi_v_override``).  ``(None, None)`` when the flag is off — the
    scalar paths stay bit-identical.  Runtime closures (EKE / Treguier /
    Visbeck kappa fields) overwrite this where they apply; the flag is meant
    for the static Redi-only recipe (DINO R1) where no adaptive κ is active.
    """
    if not bool(getattr(gm_cfg, "kappa_redi_lat_scaling", False)):
        return None, None
    evaluation = getattr(
        gm_cfg, "kappa_redi_horizontal_evaluation", "cosine_scaled")
    if evaluation not in ("cosine_scaled", "nemo_metric_literal"):
        raise ValueError(
            "unknown GMRediConfig.kappa_redi_horizontal_evaluation "
            f"{evaluation!r}; expected 'cosine_scaled' or "
            "'nemo_metric_literal'")
    if evaluation == "nemo_metric_literal":
        velocity = getattr(gm_cfg, "kappa_redi_diffusive_velocity", None)
        if velocity is None:
            raise ValueError(
                "kappa_redi_horizontal_evaluation='nemo_metric_literal' "
                "requires kappa_redi_diffusive_velocity")
        geom = ensure_geometry(grid)
        # NEMO ldf_c2d('TRA'): pUfac is formed first, then multiplied by
        # MAX(e1,e2) independently at the U and V points
        # (ldfc1d_c2d.F90:141-145).  The east/north slices use the operator's
        # cell-indexed face convention.
        pUfac = 0.5 * velocity
        kappa_u = pUfac * jnp.maximum(
            geom.dx_u[:, 1:], geom.dy_u[:, 1:])
        kappa_v = pUfac * jnp.maximum(
            geom.dx_v[1:, :], geom.dy_v[1:, :])
        return kappa_u, kappa_v
    lat = jnp.asarray(grid.lat)
    if lat.ndim != 1:
        raise ValueError(
            "GMRediConfig.kappa_redi_lat_scaling requires a regular grid "
            "with 1-D latitudes (Mercator dx = R·dλ·cos φ); this grid's "
            f"lat is {lat.ndim}-D (tripole/curvilinear) — supply the "
            "kappa field via the runtime override instead.")
    cos_lat = jnp.cos(lat)                                # (n_lat,)
    n_lon = int(getattr(grid, "n_lon"))
    kappa_T = (gm_cfg.kappa_Redi * cos_lat)[:, None] * jnp.ones(
        (1, n_lon), dtype=cos_lat.dtype)
    # cos at the TRUE v-face latitude, matching NEMO's cos(gphiv).  Carried by
    # BOTH LatLonGrid and LatLonCGridGeometry (the from-rest path); do NOT
    # rebuild it from midpoints of grid.lat -- on a stretched grid the true
    # faces are not the midpoints (DINO: 2.5e-4) -- and do NOT substitute
    # vface_zonal_cos_lat, which is the #516 pole-zeroed transport metric.
    cos_lat_v = jnp.asarray(grid.cos_lat_v)[1:]           # north face of cell j
    if bool(getattr(getattr(grid, "fold", None), "is_active", False)):
        # Tripole has NO 1-D v-face axis; its cos_lat_v is a zonal-mean
        # representative, not the real face metric.  The ndim!=1 guard above
        # does NOT catch it (tripole stores a zonal-mean 1-D lat), so refuse
        # here rather than run a fabricated ahtv.  fold.is_active is a STATIC
        # Python bool -- never inspect array VALUES here, this runs inside jit.
        raise ValueError(
            "kappa_redi_lat_scaling needs a 1-D v-face cos(lat_v); a tripolar "
            "grid has no 1-D v-face axis. Supply a 2-D kappa_Redi field.")
    kappa_v = (gm_cfg.kappa_Redi * cos_lat_v)[:, None] * jnp.ones(
        (1, n_lon), dtype=cos_lat_v.dtype)
    return kappa_T, kappa_v


def thickness_weighted_tracer_combine(t_before, t_now, t_expl, d_diss,
                                      h_before, h_now, h_after, mask,
                                      h_floor=1.0e-10):
    """NEMO thickness-weighted leap-frog tracer combine (``trazdf.F90:271-278``).

    NEMO advances tracer CONTENT, not concentration::

        e3t(Kaa)·T(Kaa) = e3t(Kbb)·T(Kbb) + 2·rdt·e3t(Kmm)·RHS

    Mapping onto this model's intermediates: ``_step_impl`` already builds the
    flux-form content update (``hT = h_now·T_now − dt·div``, then
    ``T_expl = hT / h_after``), so the RAW advective content increment is
    exactly ``h_after·T_expl − h_now·T_now`` ( = −2·rdt·div ) with no
    reweighting — matching NEMO's ``2·rdt·e3t(Kmm)·RHS_adv``, whose trends are
    each divided by e3t(Kmm) before being multiplied by it again.  The
    dissipative increment arrives as a CONCENTRATION tendency and therefore
    takes the Kmm ("now") thickness, which is the weight NEMO gives every trend
    in ``ts(:,:,:,:,Nrhs)`` — including ``tra_ldf``, which it evaluates at Kbb
    but still weights by e3t(Kmm).

    Why this matters: the bare-concentration form
    ``T(Naa) = T(Nbb) + (T_expl − T(Nnn)) + diss`` does NOT conserve tracer
    content under a moving (z-star/vvl) coordinate.  Measured on the DINO
    oracle (#1226): +8.6e-6 relative drift in globally-integrated heat over 200
    forcing-free steps, linear in step count, where NEMO drifts +3.4e-16.
    Switching to this form removes half of that leak.

    Parameters
    ----------
    t_before, t_now, t_expl : array
        Tracer at Nbb, Nnn, and the explicit (post-``_step_impl``) state.
    d_diss : array or float
        Dissipative CONCENTRATION increment from the Nbb pass.
    h_before, h_now, h_after : array
        Layer thickness at Nbb, Nnn, Naa.
    mask : array
        Wet mask; dry cells hold ``t_now`` (never 0 — writing 0 below the
        seafloor is the #480 masked-cold-cell poison).
    h_floor : float
        Division guard for dry columns.

    Returns
    -------
    array : tracer at Naa.
    """
    content = thickness_weighted_tracer_content(
        t_before, t_now, t_expl, d_diss,
        h_before, h_now, h_after)
    return jnp.where(mask > 0, content / jnp.maximum(h_after, h_floor), t_now)


def thickness_weighted_tracer_content(t_before, t_now, t_expl, d_diss,
                                      h_before, h_now, h_after):
    """Undivided NEMO ``tra_zdf`` RHS content for one tracer.

    Kept separate from :func:`thickness_weighted_tracer_combine` so the
    literal ``trazdf`` matrix can consume the source's content RHS directly;
    dividing by ``e3t(Kaa)`` and multiplying back would lose the source's
    floating-point evaluation order.
    """
    return (h_before * t_before
            + (h_after * t_expl - h_now * t_now)
            + h_now * d_diss)


def _thickness_weighted_asselin(now, before, after,
                                e3_now, e3_before, e3_after, e3_f,
                                gamma, mask):
    """NEMO thickness-weighted Robert-Asselin tracer filter (``tra_atf_qco_lf``).

    Transcribes NEMO 5.0.2 ``src/OCE/TRA/traatf_qco.F90:295-306`` — the leap-frog
    Asselin filter for tracers under the z\\* (``key_qco``) moving coordinate.
    The filter acts on the tracer CONTENT ``(e3t·T)``, NOT the concentration
    ``T``: the after-filter global content is perturbed only by the exact discrete
    time-Laplacian of content ``gamma·(C_b − 2·C_n + C_a)`` (zero when the
    flux-form advection holds the three time-level contents equal), with NO
    spurious ``e3t``-inconsistency source — unlike the plain concentration form,
    which adds an extra ``O(gamma·dη)`` content drift under a moving thickness::

        ztc_b = e3t_b·T_b ; ztc_n = e3t_n·T_n ; ztc_a = e3t_a·T_a   (:298-300)
        ztc_d = ztc_a - 2·ztc_n + ztc_b                            (:302)
        ztc_f = ztc_n + rn_atfp·ztc_d                              (:304)
        T_f   = ztc_f / e3t_f                                      (:341)

    ``e3t_f`` (:308 ``ze3t_f = e3t_0·(1 + r3t_f·tmask)``) is the thickness
    reconstructed from the *Asselin-filtered* ssh ``r3t_f`` — since z\\*
    thickness is exactly linear in ssh, this equals the thickness-weighted
    filter of ``e3t`` on wet cells, and here is supplied by
    ``compute_layer_thickness(eta_f)`` with ``eta_f`` the plain-RA-filtered ssh
    (the ``ssh_atf`` step), keeping the numerator/denominator consistent.

    Contrast the plain concentration form ``tra_atf_fix_lf`` (:227,
    ``T_f = T_n + gamma·(T_b - 2·T_n + T_a)``) used only for the linear/fixed
    free surface (``lk_linssh``), which is O(dη) non-conservative under z\\*.

    OMITTED (documented, not silently dropped): NEMO's surface-flux Asselin
    correction ``ztc_f -= zfact1·(psbc_tc − psbc_tc_b)`` at the surface level
    (traatf_qco.F90:309, + the qsr/rnf/isf analogues :313-336) cancels the
    leap-frog imprint of the EXPLICIT surface tracer flux ``sbc_tsc``.  legoESM
    applies surface T/S forcing as IMPLICIT restoring (node 20), so there is no
    explicit ``sbc_tc``/``sbc_tc_b`` content pair to difference — the term has no
    transcribable analog here and is O(gamma·2dt·Δflux) (second order for a
    restoring BC).  This routine implements the thickness-weighting (:295-341)
    only; the surface-flux correction lives with the (different) forcing path.

    ``e3_f`` is floored before the divide (masked / below-seafloor cells have
    ``e3_f == 0`` for partial-cell coords); ``mask`` re-zeros them so the floor
    never leaks into a wet result.  Sign convention: symmetric time-Laplacian
    smoothing, ``gamma = rn_atfp > 0`` (diffusive) — matches the plain-RA sign.
    """
    ztc_n = e3_now * now
    ztc_b = e3_before * before
    ztc_a = e3_after * after
    ztc_f = ztc_n + gamma * (ztc_b - 2.0 * ztc_n + ztc_a)
    e3_f_safe = jnp.maximum(e3_f, 1.0e-30)
    # Dry/below-seafloor cells CARRY the now concentration (not 0): a zeroed
    # tracer in a dry cell is stencil poison on 3-D staircase coords (#480).
    return jnp.where(mask > 0, ztc_f / e3_f_safe, now)


def _seed_centred_forcing_carry(surface_forcing, freshwater, rho_0, land_mask):
    """Seed ``state.{tau_x,tau_y}_prev``/``freshwater_eta_prev`` for
    ``barotropic_forcing_centred`` (#1226 item 3) from THIS step's forcing.

    Called (a) after the leap-frog forward-Euler start step (NEMO nit000,
    ``sbcmod.F90:568-573`` — no-restart "before := now" rule: the step-1
    ``_step_impl`` call already ran with ``tau_x_prev=None`` so its wind/emp
    was unaveraged NOW, matching NEMO's degenerate first-step average) and
    (b) after every subsequent leap-frog step, so the NEXT step's ½
    (before+now) average has a real before-value.  A plain module-level
    helper (not a closure) per the "no helper fns rebuilt inside the hot
    loop" rule — called once per step, not per-substep.
    """
    out = {}
    if surface_forcing is not None:
        _tx = getattr(surface_forcing, "tau_x", None)
        _ty = getattr(surface_forcing, "tau_y", None)
        if _tx is not None and _ty is not None:
            out["tau_x_prev"] = _tx
            out["tau_y_prev"] = _ty
    if freshwater is not None:
        out["freshwater_eta_prev"] = (
            freshwater_eta_tendency(freshwater, rho_0) * land_mask)
    return out


class LatLonCGridOceanModel:
    """Boussinesq hydrostatic ocean model on a C-grid latitude-longitude grid.

    Uses split-explicit time stepping with compact-stencil C-grid operators
    for pressure gradient and divergence.

    Parameters
    ----------
    grid : LatLonGrid
        Horizontal grid.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    config : LatLonCGridOceanConfig, optional
        Model configuration.

    Example
    -------
    >>> grid = create_latlon_grid(90, 180)
    >>> z_coord = create_ocean_z_star(50)
    >>> model = LatLonCGridOceanModel(grid, z_coord)
    >>> state = rest_state_latlon_cgrid_ocean(grid, z_coord)
    >>> state_new = model.step(state, dt=3600.0)
    """

    def __init__(
        self,
        grid: LatLonGrid,
        z_coord: OceanZStarCoordinate,
        config: LatLonCGridOceanConfig | None = None,
        *,
        iwm_forcing=None,
        _nemo_ws_test_hooks: _NEMOWSRK3TestHooks | None = None,
    ):
        if config is not None and getattr(
                getattr(config, "lateral_viscosity", None), "A_h", 0.0) is None:
            raise ValueError(
                "LatLonCGridOceanModel: lateral_viscosity.A_h is None "
                "(= derive from the mesh) but nothing resolved it. Resolve it "
                "from the mesh's narrowest WET cell before constructing the "
                "model (legoesm.ocean.state.resolution_scaled_lateral_viscosity "
                "+ wet_min_spacing), or pin a value.")
        self.z_coord = z_coord
        self._nemo_ws_test_hooks = (
            _nemo_ws_test_hooks or _NEMOWSRK3TestHooks())
        if self._nemo_ws_test_hooks.expose_momentum_stage not in (0, 1, 2, 3):
            raise ValueError(
                "expose_momentum_stage must be one of 0, 1, 2, or 3")
        _rhs_observer = self._nemo_ws_test_hooks.slow_forcing_rhs_observer
        _rhs_observer_face = (
            self._nemo_ws_test_hooks.slow_forcing_rhs_observer_face)
        if _rhs_observer_face not in ("", "u", "v"):
            raise ValueError(
                "slow_forcing_rhs_observer_face must be '', 'u', or 'v'")
        if callable(_rhs_observer) != bool(_rhs_observer_face):
            raise ValueError(
                "slow_forcing_rhs_observer and its face must be selected "
                "together")
        if self._nemo_ws_test_hooks.expose_momentum_operator_stage not in (
                2, 3):
            raise ValueError(
                "expose_momentum_operator_stage must be 2 or 3; stages 1 and "
                "3-with-lateral-mixing are served by other hooks")
        # At CONSTRUCTION: otherwise a missing stage bucket can compile and
        # score whatever the returned slots happened to hold.
        for _zad_stage, _zad_override in enumerate((
                self._nemo_ws_test_hooks.stage2_zad_operand_override,
                self._nemo_ws_test_hooks.stage3_zad_operand_override), 2):
            if _zad_override is not None and (
                    not isinstance(_zad_override, tuple)
                    or len(_zad_override) != 3):
                # A pair or bare array would score under another field's name.
                raise ValueError(
                    f"stage{_zad_stage}_zad_operand_override must be a "
                    "(w, h_u, h_v) tuple; a None slot keeps the live operand")
        _face_r3_stage = self._nemo_ws_test_hooks.expose_stage_face_r3
        if (not isinstance(_face_r3_stage, int)
                or isinstance(_face_r3_stage, bool)
                or _face_r3_stage not in (0, 1, 2, 3)):
            # A float or a bool here would index the wrong stage's ratios and
            # score them against the oracle's stage 2 under its name.
            raise ValueError(
                "expose_stage_face_r3 must be the int 0, 1, 2, or 3")
        if not isinstance(
                self._nemo_ws_test_hooks.stage2_momentum_wzv_clock_pair, bool):
            raise ValueError(
                "stage2_momentum_wzv_clock_pair must be a bool")
        _wzv_split = self._nemo_ws_test_hooks.nemo_stage_momentum_wzv_split
        if _wzv_split is not None and not isinstance(_wzv_split, bool):
            # At CONSTRUCTION: anything truthy-but-not-bool would silently
            # select the two-solve program and the walk would score one
            # compiled program under the other's name.  ``None`` is the
            # legal sentinel for "defer to the default policy" (round 163).
            raise ValueError(
                "nemo_stage_momentum_wzv_split must be a bool or None")
        if not isinstance(
                self._nemo_ws_test_hooks.stage2_wzv_velocity_form, bool):
            # At CONSTRUCTION: anything truthy would silently select NEMO's
            # other continuity call form and the walk would score one program
            # under another program's name.
            raise ValueError(
                "stage2_wzv_velocity_form must be a bool")
        if self._nemo_ws_test_hooks.expose_momentum_operator not in (
                "", "hpg", "vorticity", "advection", "keg", "zad", "after_hpg", "after_vor", "after_keg", "after_zad"):
            raise ValueError(
                "expose_momentum_operator must name a component or after_* "
                "accumulator boundary; got "
                f"{self._nemo_ws_test_hooks.expose_momentum_operator!r}")
        if self._nemo_ws_test_hooks.expose_tracer_stage1_boundary not in (
                "", "after_advection", "after_sbc"):
            raise ValueError(
                "expose_tracer_stage1_boundary must be '', "
                "'after_advection', or 'after_sbc'")
        # Dispatch hardening at CONSTRUCTION, not at the substitution site:
        # the returned u/v carry ONE exposed frame, and the substitutions run
        # in a fixed order, so two momentum-RHS exposures at once would hand
        # the caller the later one under the earlier one's name.  A gate that
        # silently scores the wrong frame is worse than one that refuses.
        _stage3_rhs_hook = self._nemo_ws_test_hooks.expose_stage3_momentum_rhs
        if _stage3_rhs_hook not in ("", "pre_ldf", "post_ldf"):
            raise ValueError(
                "expose_stage3_momentum_rhs must be '', 'pre_ldf', or "
                f"'post_ldf'; got {_stage3_rhs_hook!r}")
        _stage1_slots = tuple(
            (_name, getattr(self._nemo_ws_test_hooks, _name))
            for _name in ("expose_stage1_momentum_rhs",
                          "expose_stage1_raw_momentum"))
        for _name, _flag in _stage1_slots:
            if not isinstance(_flag, bool):
                # At CONSTRUCTION: anything truthy-but-not-bool would select
                # the exposure silently and the walk would score one stage-1
                # boundary under the other's name.
                raise ValueError(f"{_name} must be a bool")
        _stage1_split = (
            self._nemo_ws_test_hooks.expose_stage1_momentum_rhs_split)
        _tr_operand = (
            self._nemo_ws_test_hooks.momentum_transport_stage1_operand)
        if _tr_operand not in _STAGE1_TRANSPORT_OPERAND_ARMS:
            # At CONSTRUCTION, like its sibling below: validating this inside
            # the stage branch would silently accept a typo on any card whose
            # transport reconcile is off or whose integrator is not WS-RK3,
            # and the walk would score the production path under the arm's
            # name (review finding).
            raise ValueError(
                "momentum_transport_stage1_operand must be one of "
                f"{_STAGE1_TRANSPORT_OPERAND_ARMS!r}; got {_tr_operand!r}")
        if _stage1_split not in _STAGE1_SPLIT_ARMS:
            raise ValueError(
                "expose_stage1_momentum_rhs_split must be one of "
                f"{_STAGE1_SPLIT_ARMS!r}; got {_stage1_split!r}")
        _stage1_selected = [_name for _name, _flag in _stage1_slots if _flag]
        if _stage1_split:
            # It writes the SAME returned u/v slots as the two bools above.
            _stage1_selected.append("expose_stage1_momentum_rhs_split")
        if _stage1_selected and (
                len(_stage1_selected) > 1
                or bool(self._nemo_ws_test_hooks.expose_momentum_stage)
                or self._nemo_ws_test_hooks.expose_stage2_raw_momentum
                or self._nemo_ws_test_hooks.expose_stage2_momentum_rhs
                or bool(self._nemo_ws_test_hooks.expose_momentum_operator)
                or bool(_stage3_rhs_hook)
                # These two also write the returned u/v slots, and they are
                # substituted EARLIER, so a stage-1 exposure would silently
                # win and the walk would score the stage-1 frame under the
                # transport's name.
                or bool(self._nemo_ws_test_hooks
                        .expose_tracer_transport_stage)
                or bool(self._nemo_ws_test_hooks
                        .expose_stage1_transport_operand)
                # The exposure publishes the PRODUCTION right-hand side, so
                # selecting it together with the override would publish the
                # array the step did NOT use.
                or (self._nemo_ws_test_hooks.stage1_momentum_rhs_override
                    is not None)):
            raise ValueError(
                "expose_stage1_momentum_rhs / expose_stage1_raw_momentum "
                "cannot be combined with another momentum exposure or with "
                "stage1_momentum_rhs_override: they share the returned u/v "
                "slots")
        if _stage3_rhs_hook and (
                self._nemo_ws_test_hooks.expose_stage2_momentum_rhs
                or self._nemo_ws_test_hooks.expose_stage2_raw_momentum
                or self._nemo_ws_test_hooks.expose_momentum_stage
                or self._nemo_ws_test_hooks.expose_momentum_operator):
            raise ValueError(
                "expose_stage3_momentum_rhs cannot be combined with another "
                "momentum exposure: they share the returned u/v slots")
        _stage3_raw_hook = (
            self._nemo_ws_test_hooks.expose_stage3_raw_momentum)
        if not isinstance(_stage3_raw_hook, bool):
            raise ValueError("expose_stage3_raw_momentum must be a bool")
        if _stage3_raw_hook and (
                _stage3_rhs_hook
                or self._nemo_ws_test_hooks.expose_stage2_momentum_rhs
                or self._nemo_ws_test_hooks.expose_stage2_raw_momentum
                or self._nemo_ws_test_hooks.expose_momentum_stage
                or self._nemo_ws_test_hooks.expose_momentum_operator):
            raise ValueError(
                "expose_stage3_raw_momentum cannot be combined with another "
                "momentum exposure: they share the returned u/v slots")
        _zdf_momentum_observer = (
            self._nemo_ws_test_hooks.zdf_momentum_observer)
        if (_zdf_momentum_observer is not None
                and not callable(_zdf_momentum_observer)):
            raise ValueError("zdf_momentum_observer must be callable or None")
        _config_input = config or LatLonCGridOceanConfig.from_flat()
        _oracle_endpoint_diagnostic_eos_bypass = bool(
            (self._nemo_ws_test_hooks.expose_stage1_wzv
             or self._nemo_ws_test_hooks.expose_tracer_stage1_boundary)
            and self._nemo_ws_test_hooks.external_mode_result_override is not None
            and _config_input.eos == "nemo_eos80"
            and _config_input.eos_depth == "geometric"
        )
        _config_for_validation = (
            _config_input._replace(eos_depth="insitu")
            if _oracle_endpoint_diagnostic_eos_bypass else _config_input
        )
        self.config = self._validate_config(_config_for_validation)
        if _oracle_endpoint_diagnostic_eos_bypass:
            self.config = self.config._replace(eos_depth="geometric")
        if (self._nemo_ws_test_hooks
                .stage3_advection_content_override is not None):
            if self.config.tracer_time_integrator != "rk3_ws":
                raise ValueError(
                    "stage-3 FCT pair hook requires "
                    "tracer_time_integrator='rk3_ws'")
        _process_trace = self._nemo_ws_test_hooks.tracer_process_trace
        if (self._nemo_ws_test_hooks.stage3_qsr_stretch_override is not None
                and _process_trace is None):
            raise ValueError(
                "stage3_qsr_stretch_override requires tracer_process_trace")
        if (self._nemo_ws_test_hooks.vertical_solve_trace
                and _process_trace is None):
            raise ValueError(
                "vertical_solve_trace requires tracer_process_trace")
        if (self._nemo_ws_test_hooks.tracer_process_branch_activity
                and _process_trace is None):
            raise ValueError(
                "tracer diagnostic hook requires "
                "tracer_process_trace")
        if (self._nemo_ws_test_hooks.tracer_ldf_diagnostics is not None
                and (self.config.outer_integrator != "forward_euler"
                     or self.config.tracer_time_integrator != "rk3_ws"
                     or self.config.gm_redi is None)):
            raise ValueError(
                "tracer_ldf_diagnostics requires the forward-Euler WS-RK3 "
                "GM/Redi production path")
        if _process_trace is not None:
            if not isinstance(_process_trace, tuple) or len(
                    _process_trace) not in (0, 4):
                raise ValueError(
                    "tracer_process_trace must be () or (j, i, k, delta)")
            if (self.config.outer_integrator != "forward_euler"
                    or self.config.tracer_time_integrator != "rk3_ws"
                    or self.config.gm_redi is None
                    or self.config.physics.shortwave_penetration.scheme
                    != "nemo_qsr_2bd"
                    or self.config.physics.lateral_mixing.scheme != "none"
                    or self.config.use_conservation_fixer):
                raise ValueError(
                    "tracer_process_trace is defined only for the resolved "
                    "GYRE WS-RK3/QSR/GM-Redi process program")
        # The LDF route is production behavior for WS with a configured
        # GM/Redi operator; it has no private selector and therefore needs no
        # construction-time check beyond the card's ordinary validation.
        # The compiled program accumulates this rate into tracer Krhs before
        # tra_zdf consumes it (stprk3_stg.F90:950-965).

        # Convert LatLonGrid -> LatLonCGridGeometry once at construction.
        # All downstream operators see the enriched geometry with per-cell
        # metric arrays.  For a plain LatLonGrid this is a no-op on field
        # access (legacy fields are identical); for a tripolar grid the
        # geometry carries fold descriptor and rotation angles.
        # ``metric_convention`` (#1226) is validated above, so the raise on
        # an unknown value happens before this call.
        # THE SILENT-IGNORE CASE, refused rather than documented away.
        # ``ensure_geometry`` passes a pre-built LatLonCGridGeometry through
        # UNCHANGED, so a non-default placement requested on the config would
        # do nothing at all -- the model would run the default convention while
        # its own config said otherwise.  A supplied geometry is therefore
        # CHECKED against the request, by the one identity that separates the
        # conventions without needing the face latitudes: under "cell_average"
        # the interior ``f_v`` IS the mean of the two adjacent ``f_T`` rows, to
        # the arithmetic that built it.
        if (self.config.coriolis_placement != "cell_average"
                and hasattr(grid, "f_v") and hasattr(grid, "f_T")):
            _fT = jnp.asarray(grid.f_T)
            _fv_int = jnp.asarray(grid.f_v)[1:-1]
            _gap = float(jnp.max(jnp.abs(
                _fv_int - 0.5 * (_fT[:-1] + _fT[1:]))))
            _scale = float(jnp.max(jnp.abs(_fT))) or 1.0
            # AND ONLY WHEN THE TWO CONVENTIONS ACTUALLY DIFFER ON THIS GRID.
            # The cell average equals the face value EXACTLY wherever f is
            # LINEAR between the rows -- every beta-plane and every f-plane.
            # There the setting is not being ignored, it is indistinguishable,
            # and raising would abort a perfectly valid run (adversarial
            # review, round 2, measured: beta-plane and f-plane both gap 0.0
            # and both tripped the guard).  Curvature of f along the rows is
            # the discriminator, and on a lat-lon grid it is exactly the second
            # difference of f_T.
            _curv = (float(jnp.max(jnp.abs(_fT[2:] - 2.0 * _fT[1:-1] + _fT[:-2])))
                     if _fT.shape[0] >= 3 else 0.0)
            _conventions_differ = _curv > 1e-12 * _scale
            if _conventions_differ and _gap <= 1e-12 * _scale:
                raise ValueError(
                    "coriolis_placement="
                    f"{self.config.coriolis_placement!r} was requested, but "
                    "the grid handed to this model is an ALREADY-BUILT "
                    "LatLonCGridGeometry whose f_v is the cell average (max "
                    f"departure {_gap:.3e} <= {1e-12 * _scale:.3e}). "
                    "ensure_geometry passes a pre-built geometry through "
                    "unchanged, so this setting would be silently ignored "
                    f"(f_T curvature {_curv:.3e} shows the two conventions DO "
                    "differ on this grid, so this is a real no-op and not a "
                    "beta-plane coincidence). "
                    "Select the placement where the geometry is built "
                    "(create_latlon_geometry / ensure_geometry / "
                    "bridge_nemo_to_legoesm_topo)."
                )
        self.grid = ensure_geometry(
            grid, metric_convention=self.config.metric_convention,
            vface_zonal_metric_evaluation=(
                self.config.vface_zonal_metric_evaluation),
            coriolis_placement=self.config.coriolis_placement)
        # Push the meridionally-FLAT (Oceananigans `Flat`-y) mode to the grid-
        # operators backend PROCESS-GLOBAL (same pattern as the halo backend).
        # CONSTRAINT: this is process-global, so it assumes ONE lat-lon ocean model
        # per process — constructing a second model with a different
        # ``meridionally_flat`` flips the global for BOTH (the operators read it at
        # trace time, so a model that recompiles after the flip silently bakes in
        # the other model's setting).  The unconditional set keeps the global in
        # sync with the MOST-RECENTLY-CONSTRUCTED model; for the normal single-model
        # run this is correct, and default False ⇒ bit-identical.  NOTE: the face
        # masks are built at STATE construction (before the model), so an experiment
        # that wants flat-y masks must ALSO call ``set_meridionally_flat(True)``
        # before building the rest state (partial-cell steps rebuild masks from
        # ``z_coord.is_active`` and so honour the flag regardless; non-partial-cell
        # paths consume the stored mask and need the pre-set).
        from legoesm.grids.halo_latlon import set_meridionally_flat
        set_meridionally_flat(bool(getattr(self.config, "meridionally_flat", False)))
        # Wide-halo barotropic refuses tripolar folds — fail at construction
        # with the config knob named, not at the first traced step deep in
        # widen_cgrid_geometry_band (codex: CLI accepted a tripole + wide
        # combination that only failed mid-run).
        if self.config.barotropic.barotropic_wide_halo:
            from legoesm.ocean.dynamics.latlon_cgrid_operators import (
                is_tripolar,
            )
            if is_tripolar(self.grid):
                raise ValueError(
                    "barotropic_wide_halo=True is not supported on tripolar "
                    "grids (the fold row needs a permuted, sign-flipped wide "
                    "exchange — follow-up); disable the wide-halo barotropic "
                    "for eORCA/tripole runs.")
        # GEOMETRIC EKE closure (Torres et al. 2025) needs the regular-grid
        # B_T operator (flux_divergence_viscosity_cgrid raises on tripolar);
        # fail at construction, not at the first traced step.
        _gm = self.config.gm_redi
        if _gm is not None:
            validate_redi_coefficient(_gm)
        if (_gm is not None and getattr(_gm, "eke", None) is not None
                and _gm.eke.closure == "geometric"):
            from legoesm.ocean.dynamics.latlon_cgrid_operators import is_tripolar
            if is_tripolar(self.grid):
                raise ValueError(
                    "EKEConfig.closure='geometric' is not supported on "
                    "tripolar grids (the B_T shear-production operator "
                    "flux_divergence_viscosity_cgrid is regular-lat-lon "
                    "only). Use a regular lat-lon grid or extend the "
                    "operator."
                )
        self._cfl_checked = False
        # Internal wave-driven mixing (zdfiwm) static 2-D forcing maps
        # (IWMForcing of de Lavergne power/decay-scale fields on THIS
        # grid), captured as closure constants by the jitted step.  None
        # with iwm.enabled=True ⇒ the uniform constant-power fallback
        # from the IWMConfig scalars.
        self._iwm_forcing = iwm_forcing
        _vmix_cfg_init = (self.config.physics.vertical_mixing
                          if self.config.physics is not None else None)
        if (iwm_forcing is not None
                and (_vmix_cfg_init is None
                     or getattr(_vmix_cfg_init, "iwm", None) is None
                     or not _vmix_cfg_init.iwm.enabled)):
            raise ValueError(
                "iwm_forcing was supplied but "
                "physics.vertical_mixing.iwm.enabled is not True — the maps "
                "would be silently ignored.")
        if (_vmix_cfg_init is not None
                and getattr(_vmix_cfg_init, "iwm", None) is not None
                and _vmix_cfg_init.iwm.enabled
                and not getattr(self.config, "implicit_vertical_mixing", False)):
            raise ValueError(
                "vertical_mixing.iwm.enabled=True requires "
                "implicit_vertical_mixing=True (zdfiwm contributes to the "
                "implicit avt/avm profiles; the explicit path cannot apply "
                "its momentum part).")
        if (_vmix_cfg_init is not None
                and getattr(_vmix_cfg_init, "ddm", None) is not None
                and _vmix_cfg_init.ddm.enabled
                and not getattr(self.config, "implicit_vertical_mixing", False)):
            raise ValueError(
                "vertical_mixing.ddm.enabled=True requires "
                "implicit_vertical_mixing=True (zdfddm routes the SALINITY "
                "solve through a separate diffusivity K_v + (avs - avt) in the "
                "implicit path; the explicit / shared-K pair path cannot carry "
                "avs != avt — codex r2).")
        # Static rigid-lid data (islands, basis, depths), built eagerly from the
        # first concrete state (host-side flood-fill).  None until built.
        self.rigid_lid_data = None
        # Source (land_mask, H_bathy) the rigid-lid cache was built from — keyed
        # exactly like _vertex_mask_src so a reused model instance can't serve
        # stale island data for a CHANGED topology (codex round-1).
        self._rigid_lid_src = None
        # Build-once vertex mask (land-mask-derived, constant per run).
        # Computing it per step paid an N-S halo exchange inside the
        # traced tendencies (x2 sites) because state.land_mask is a
        # step-input tracer even though its VALUE never changes (halo
        # census job 8474554).  Filled eagerly by step()'s Python body
        # on the first call; threaded into the tendencies as a closure
        # constant.  Callers that trace _step_impl directly without ever
        # calling step() keep the in-graph fallback (correct, no win).
        # STALENESS CONTRACT: like the geometry itself, this assumes the
        # land mask is fixed after construction (replace_land_mask is a
        # construction-stage tool; a mid-run mask swap requires a new
        # model instance).
        self._vertex_mask = None
        self._vertex_mask_src = None

        # Surface-forcing IMPLICIT routing (Veros placement): when enabled, the
        # combined physics is built with surface_forcing.scheme="none" so the
        # T*/S* restoring is NOT summed into the explicit dT_dt; a SEPARATE
        # restoring function (``_surface_tracer_forcing_fn``) supplies the
        # restoring RATE, which the tendency builder routes onto
        # ``tendencies.surface_tracer_forcing`` (together with the withheld
        # q_net / shortwave heat) for the backward-Euler implicit application.
        self._surface_tracer_forcing_fn = None
        if self.config.physics is not None:
            physics_for_combined = self.config.physics
            if self.config.surface_forcing_implicit:
                from legoesm.ocean.physics.surface_forcing.integration import (
                    make_surface_forcing_physics,
                )
                _sf_cfg = self.config.physics.surface_forcing
                # Build the restoring rate function from the REAL surface_forcing
                # config (verbatim reuse — no duplicated restoring numerics).
                self._surface_tracer_forcing_fn = make_surface_forcing_physics(_sf_cfg)
                # Strip the restoring from the explicit combined physics so it is
                # not double-counted (summed into dT_dt) — it is applied
                # implicitly instead.
                _sf_none = _sf_cfg._replace(scheme="none")
                physics_for_combined = self.config.physics._replace(
                    surface_forcing=_sf_none,
                )
            # MED-2 latitude-dependent constant background (codex batch2
            # BLOCKER): the Gregg (2003) lat/N-scaled field can only be
            # assembled by the IMPLICIT fallback (compute_vertical_K_profiles
            # needs the column latitudes + N²).  The explicit constant physics
            # fn cannot represent it (constant_vertical_mixing raises,
            # fail-loud), and any spatially-constant K/A it surfaced would
            # take the fast path in _apply_implicit_vertical_mixing, which
            # adds the model floors the REPLACE semantics forbids.  Strip the
            # vmix scheme from the EXPLICIT composition (its tendencies are
            # zero under implicit mixing anyway) and let the implicit solve
            # recompute K/A via the fallback — the same pattern as the
            # surface_forcing_implicit strip above.  Explicit-mixing configs
            # (implicit_vertical_mixing=False) keep the loud ValueError.
            if (self.config.implicit_vertical_mixing
                    and self._lat_dependent_constant_vmix()):
                physics_for_combined = physics_for_combined._replace(
                    vertical_mixing=physics_for_combined.vertical_mixing
                    ._replace(scheme="none"))
            self._physics_fn = make_ocean_physics(
                physics_for_combined,
                apply_vertical_diffusion=not self.config.implicit_vertical_mixing,
                # The CARD's NEMO &nameos coefficients reach the zdfevd
                # trigger's bn2 (decision 94): without this the trigger built
                # alpha/beta from NemoSEOSConfig()'s DINO defaults whatever
                # fluid the card actually runs.
                seos_cfg=self.config.eos_nemo_seos,
            )
        else:
            self._physics_fn = None

    def _lat_dependent_constant_vmix(self) -> bool:
        """Static predicate: the MED-2 latitude-dependent CONSTANT background
        is selected (``vertical_mixing.scheme="constant"`` with
        ``constant.lat_dependent=True`` on the physics config).

        Centralised so the explicit-composition strip (``__init__``) and the
        implicit-solve fallback force (``_apply_implicit_vertical_mixing``)
        can never drift apart.  Pure Python on static config — jit-safe.
        """
        _pc = self.config.physics
        return (_pc is not None
                and _pc.vertical_mixing.scheme == "constant"
                and bool(getattr(_pc.vertical_mixing.constant,
                                 "lat_dependent", False)))

    def _ensure_vertex_mask(self, state) -> None:
        """Fill (or refresh) the vertex-mask cache from CONCRETE state.

        No-op on traced state (a jitted caller wrapping ``step`` traces
        this Python body once with tracers — the in-graph fallback then
        applies).  The cache is KEYED on the source land mask (codex
        review MAJOR: an unkeyed cache silently served state A's mask to
        state B on model reuse): identity fast-path, then a host value
        compare — a different mask REFRESHES the cache.
        """
        import jax as _jax

        m = state.land_mask.data
        if isinstance(m, _jax.core.Tracer):
            return
        if self._vertex_mask is not None:
            src = self._vertex_mask_src
            if m is src:
                return
            import numpy as _np
            if (src is not None and m.shape == src.shape
                    and bool(_np.array_equal(_np.asarray(m),
                                             _np.asarray(src)))):
                self._vertex_mask_src = m   # adopt new identity, same value
                return
            # A DIFFERENT mask cannot be served by refreshing this
            # attribute: the already-compiled ``_step_jitted`` baked the
            # old mask as a closure CONSTANT (``self`` is a static
            # argument — the jit cache would silently reuse the stale
            # executable; codex round-2).  Enforce the per-instance
            # contract mechanically instead of documenting it.
            raise ValueError(
                "LatLonCGridOceanModel: the land mask changed after the "
                "first step on this model instance.  The vertex-mask "
                "cache (and the compiled step that captured it) are "
                "built once per model — construct a NEW model for a "
                "different mask (replace_land_mask is a construction-"
                "stage tool)."
            )
        from legoesm.grids.operators_latlon_cgrid import compute_vertex_mask

        self._vertex_mask = compute_vertex_mask(m, grid=self.grid)
        self._vertex_mask_src = m

    def prime_step_caches(self, state) -> None:
        """Eagerly fill build-once step caches from CONCRETE state.

        Public hook for drivers that integrate via ``_step_impl``
        directly inside an outer ``lax.scan``/``jax.jit`` (the public
        ``step`` shim does this automatically): call ONCE with the
        concrete initial state BEFORE building/tracing the scan, so the
        traced body captures the build-once caches as constants instead
        of re-deriving them every step.  Two caches:

        * the land-mask-derived vertex mask (its N-S halo exchange), and
        * (rigid-lid solver only) the host-side island / streamfunction
          decomposition (numpy flood-fill — impossible to build from a
          traced state, so it MUST be warmed here from the concrete IC).

        Safe to call multiple times and with traced state (no-op): both
        warms are tracer-guarded.  A cold rigid-lid cache then left to a
        traced first ``_step_impl`` step raises an actionable error.
        """
        self._ensure_vertex_mask(state)
        # Warm the rigid-lid island cache only when EVERY build input is
        # concrete: a partially-traced state (e.g. concrete H_bathy + traced
        # land_mask) would otherwise reach the cold host-side flood-fill and
        # raise, breaking the "no-op on traced state" contract (codex round-2).
        if self.config.barotropic.barotropic_solver == "rigid_lid":
            _rl_inputs = (state.H_bathy.data, state.land_mask.data,
                          state.u_mask.data, state.v_mask.data)
            if not any(isinstance(a, jax.core.Tracer) for a in _rl_inputs):
                self._ensure_rigid_lid_data(state)

    def _ensure_rigid_lid_data(self, state):
        """Build + cache the static rigid-lid data from a CONCRETE state.

        The island flood-fill is host-side (numpy), so this must run eagerly on
        concrete bathymetry/masks: ``integrate_scan`` pre-builds it before the
        (jitted) scan via ``seed_scan_carry``, and the eager ``step`` shim
        pre-builds it from its concrete input state (so ``model.step`` /
        ``model.integrate`` loops work too) — in both paths the captured data is
        a compile-time constant.  Only a direct ``_step_impl`` call inside a
        user trace builds it lazily, and there the state must already be warmed
        (else the actionable ``TracerArrayConversionError`` below fires).

        ``ensure_compile_time_eval``: when the lazy build instead happens
        inside a USER's jit trace (e.g. ``jax.jit(value_and_grad(loss))``
        whose loss takes the first-ever rigid-lid step, with the state a
        closure constant), the masks are concrete but omnistaging would
        stage the ``jnp`` parts of the build and CACHE TRACERS on the model
        — a leaked-tracer error on reuse (2026-06-11 differentiability
        audit). Forcing compile-time eval builds concrete arrays in that
        context too. If the masks themselves are traced (state passed as a
        jit argument on first use), the host-side ``np.asarray`` in the
        builder raises ``TracerArrayConversionError`` — re-raised with an
        actionable message.
        """
        # Fail fast BEFORE building anything: build_rigid_lid_data runs the
        # global streamfunction-basis solves + island line integrals, which are
        # single-rank only (rank-local CG dots, domain-wide jnp.sum).  This path
        # is reached EAGERLY from seed_scan_carry (integrate/integrate_scan), so
        # the guard fires before the jitted scan on those entry points.
        from legoesm.ocean.dynamics.rigid_lid_latlon_cgrid import (
            assert_rigid_lid_single_rank,
        )
        assert_rigid_lid_single_rank()
        lm = state.land_mask.data
        Hb = state.H_bathy.data
        if self.rigid_lid_data is not None:
            # Topology-key guard (mirrors _ensure_vertex_mask): the island
            # decomposition is baked into the compiled step as a constant, so a
            # reused model must NOT silently serve stale data for a different
            # bathymetry/mask.  The hot loop passes the SAME concrete arrays
            # (identity fast-path) or TRACED arrays (cache already built — skip,
            # can't host-compare); only a genuinely different CONCRETE topology
            # trips the value compare.  u_mask/v_mask are deterministic
            # functions of land_mask, so keying on (land_mask, H_bathy) covers
            # the full island input set.
            if (isinstance(lm, jax.core.Tracer)
                    or isinstance(Hb, jax.core.Tracer)):
                return self.rigid_lid_data
            src_lm, src_Hb = self._rigid_lid_src
            if lm is src_lm and Hb is src_Hb:
                return self.rigid_lid_data
            import numpy as _np
            if (src_lm is not None and lm.shape == src_lm.shape
                    and Hb.shape == src_Hb.shape
                    and bool(_np.array_equal(_np.asarray(lm), _np.asarray(src_lm)))
                    and bool(_np.array_equal(_np.asarray(Hb), _np.asarray(src_Hb)))):
                self._rigid_lid_src = (lm, Hb)   # adopt new identity, same value
                return self.rigid_lid_data
            raise ValueError(
                "LatLonCGridOceanModel: the bathymetry/land mask changed after "
                "the first rigid-lid step on this model instance.  The island "
                "decomposition (and the compiled step that captured it) are "
                "built once per model — construct a NEW model for a different "
                "topology.")
        from legoesm.ocean.dynamics.rigid_lid_islands import build_rigid_lid_data
        try:
            with jax.ensure_compile_time_eval():
                self.rigid_lid_data = build_rigid_lid_data(
                    Hb, lm, state.u_mask.data, state.v_mask.data,
                    self.config, self.grid, periodic_x=True,
                )
                self._rigid_lid_src = (lm, Hb)
        except jax.errors.TracerArrayConversionError as e:
            raise RuntimeError(
                "rigid-lid island decomposition must be built from CONCRETE "
                "bathymetry/masks, but the state reaching the first rigid-lid "
                "step is traced (it was passed as an argument into a "
                "jit/grad-transformed function). Warm the cache once before "
                "transforming: call model.step(state, dt) eagerly, or "
                "model._ensure_rigid_lid_data(state) on the concrete initial "
                "state."
            ) from e
        return self.rigid_lid_data

    @staticmethod
    def _validate_config(
        config: LatLonCGridOceanConfig,
    ) -> LatLonCGridOceanConfig:
        """Resolve conditional defaults, then validate configuration ranges."""
        nonnegative = {
            "A_h": config.lateral_viscosity.A_h,
            "B_h": config.lateral_viscosity.B_h,
            "K_h": config.K_h,
            "A_v": config.A_v,
            "K_v": config.K_v,
            "hyperdiff_coeff": config.hyperdiff_coeff,
            "barotropic_diffusion_alpha": config.barotropic.barotropic_diffusion_alpha,
            "bbl_gamma_s": config.bbl_gamma_s,
        }
        for name, value in nonnegative.items():
            if value is None:
                # A_h=None means "derive from the mesh"; comparing it to 0.0
                # would raise a bare TypeError instead of saying what to do.
                raise ValueError(
                    f"{name} is None (= derive from the mesh) but nothing "
                    "resolved it. Resolve it from the mesh's narrowest WET "
                    "cell before validating or running "
                    "(legoesm.ocean.state.resolution_scaled_lateral_viscosity "
                    "+ wet_min_spacing), or pin a value.")
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value!r}")

        # ``config.constants`` and ``config.physics.constants`` are ONE set.
        # ``from_flat`` / ``replace_flat`` propagate ``config.constants`` into
        # the physics pipeline automatically; a config built through the RAW
        # NamedTuple constructor bypasses that routing, so catch the divergence
        # here rather than silently running the momentum path on one g and
        # compute_N2 / KPP / TKE on another (the #1226 N^2 residual).
        # SCOPE: this pairing only. ``FluxFeedbackConfig`` deliberately owns a
        # SEPARATE c_sw/rho_0 (the Veros surface-forcing cp_0, documented on
        # that config) and nothing propagates ``constants`` into it -- do not
        # read this check as "the model has exactly one c_sw".
        # ``constants_equal(...) is False`` (a PROVEN difference), never a bare
        # ``!=``: a traced constant would otherwise raise
        # TracerBoolConversionError here instead of this domain error -- the
        # same defect the from_flat/replace_flat routing had.
        _phys = config.physics
        if (_phys is not None and getattr(_phys, "constants", None) is not None
                and constants_equal(_phys.constants, config.constants) is False):
            raise ValueError(
                "ocean physical constants disagree between the model config "
                f"({config.constants}) and its physics pipeline "
                f"({_phys.constants}). Build the config with "
                "LatLonCGridOceanConfig.from_flat(...) (which routes the flat "
                "g=/rho_0=/omega=/c_sw=/R_earth= kwargs into config.constants "
                "and propagates them to physics) instead of the raw "
                "constructor, or pass matching ConstantsConfig values."
            )
        if _phys is not None and _phys.shortwave_penetration is not None:
            from legoesm.ocean.physics.shortwave_penetration import (
                SHORTWAVE_PENETRATION_SCHEMES,
            )
            _shortwave_scheme = _phys.shortwave_penetration.scheme
            if _shortwave_scheme not in SHORTWAVE_PENETRATION_SCHEMES:
                raise ValueError(
                    "physics.shortwave_penetration.scheme must be one of "
                    f"{SHORTWAVE_PENETRATION_SCHEMES}, got "
                    f"{_shortwave_scheme!r}")

        # #1226: T/u-face metric convention dispatch -- raise on an unknown
        # value rather than silently falling through to create_latlon_geometry's
        # own raise deep inside ensure_geometry (fail at config-validation time,
        # before any grid conversion work happens).
        if config.metric_convention not in ("exact", "nemo_isotropic"):
            raise ValueError(
                "metric_convention must be 'exact' or 'nemo_isotropic', got "
                f"{config.metric_convention!r}"
            )
        if config.vface_zonal_metric_evaluation not in (
                "legacy_tracer_midpoint", "nemo_vpoint"):
            raise ValueError(
                "vface_zonal_metric_evaluation must be "
                "'legacy_tracer_midpoint' or 'nemo_vpoint', got "
                f"{config.vface_zonal_metric_evaluation!r}"
            )

        # #1455: the vertex-Coriolis placement, same dispatch pattern.
        if config.coriolis_placement not in ("cell_average", "face_latitude"):
            raise ValueError(
                "coriolis_placement must be 'cell_average' or "
                f"'face_latitude', got {config.coriolis_placement!r}"
            )


        # Lateral mixing on the lat-lon C-grid is a DYNAMICS-level concern:
        # horizontal viscosity via config.lateral_viscosity.A_h/config.lateral_viscosity.B_h, GM/Redi via the
        # top-level config.gm_redi field (applied in the model step). The
        # physics-pathway lateral-mixing factory (config.physics.lateral_mixing)
        # is cubed-sphere-only — harmonic/biharmonic crash on lat-lon array
        # shapes and gm_redi raises TypeError — so any non-"none" scheme there
        # is a mis-wiring that would fail (or silently no-op via the probe) at
        # runtime. Reject it at construction with a clear message.
        if (config.physics is not None
                and config.physics.lateral_mixing.scheme != "none"):
            raise ValueError(
                "On the lat-lon C-grid, config.physics.lateral_mixing.scheme="
                f"{config.physics.lateral_mixing.scheme!r} is unsupported (the "
                "physics lateral-mixing factory is cubed-sphere-only). Set "
                "horizontal viscosity via config.lateral_viscosity.A_h / config.lateral_viscosity.B_h and GM/Redi "
                "via the top-level config.gm_redi; keep "
                "physics.lateral_mixing.scheme='none'."
            )

        # Dispatch hardening: the equilibrium-tide body force is applied inside
        # the SPLIT-EXPLICIT barotropic substeps (barotropic_substeps_latlon_cgrid,
        # the `else` branch of the barotropic-solver dispatch). The rigid-lid /
        # implicit-CN / unsplit solvers route AROUND that call, so an enabled tide
        # there would be a SILENT no-op — reject it LOUDLY at construction.
        _tf = getattr(config, "tidal_forcing", None)
        if _tf is not None and _tf.enabled:
            _bsolver = config.barotropic.barotropic_solver
            if _bsolver in ("rigid_lid", "implicit_cn", "implicit_unsplit"):
                raise ValueError(
                    f"tidal_forcing.enabled=True is not supported with "
                    f"barotropic_solver={_bsolver!r}: the equilibrium-tide body "
                    f"force is applied in the split-explicit barotropic substeps. "
                    f"Use barotropic_solver='explicit_substep' (the default).")

        # Meridionally-FLAT (Oceananigans `Flat`-y, ∂/∂y≡0) is wired ONLY into the
        # operators built via gradient_y_cgrid / divergence_cgrid (PGF, KE-gradient,
        # tracer advection, the scalar Laplacian, the vector-Laplacian viscosity)
        # PLUS the flux-form momentum advection's meridional flux.  Operators that
        # own their OWN meridional stencil are NOT flat-aware: the flux-divergence
        # lateral viscosity, biharmonic, GM/Redi, and the lateral-friction schemes.
        # Reject those combinations LOUDLY so a config is never silently Flat for
        # some terms and 3-D for others (the 2-D x–z oracle has none of them).
        if getattr(config, "meridionally_flat", False):
            _ungated = []
            _visc_on = (config.lateral_viscosity.A_h > 0.0 or config.lateral_viscosity.B_h > 0.0
                        or getattr(config.lateral_viscosity, "C_smag", 0.0) > 0.0)
            if _visc_on and config.lateral_viscosity_operator == "flux_divergence":
                _ungated.append(
                    'A_h/B_h/C_smag>0 with lateral_viscosity_operator='
                    '"flux_divergence" (use "vector_laplacian", which IS flat-aware)')
            if config.gm_redi is not None:
                _ungated.append("gm_redi is not None")
            if getattr(config, "lateral_friction_scheme", "none") != "none":
                _ungated.append(
                    f"lateral_friction_scheme={config.lateral_friction_scheme!r}")
            if _ungated:
                raise ValueError(
                    "meridionally_flat=True (Oceananigans Flat-y, ∂/∂y≡0) is "
                    "incompatible with meridional operators that are not flat-aware: "
                    + "; ".join(_ungated) + ". These keep live ∂/∂y terms, making "
                    "the model neither the 2-D x–z oracle nor a consistent 3-D run. "
                    "Disable them or use a flat-aware alternative.")

        if config.barotropic.n_barotropic_substeps < 1:
            raise ValueError(
                f"n_barotropic_substeps must be >= 1, got "
                f"{config.barotropic.n_barotropic_substeps!r}",
            )
        if config.barotropic.barotropic_wide_halo_chunk < 0:
            raise ValueError(
                f"barotropic_wide_halo_chunk must be >= 0 (0 = auto), got "
                f"{config.barotropic.barotropic_wide_halo_chunk!r}",
            )
        if (config.barotropic.barotropic_wide_halo
                and config.barotropic.barotropic_solver != "explicit_substep"):
            raise ValueError(
                "barotropic_wide_halo=True requires "
                "barotropic_solver='explicit_substep' (the wide-halo path "
                "replaces the substep loop's per-substep exchanges), got "
                f"{config.barotropic.barotropic_solver!r}",
            )
        if (config.barotropic.barotropic_wide_halo
                and not config.barotropic.barotropic_local_subcycle_clamp):
            raise ValueError(
                "barotropic_wide_halo=True requires "
                "barotropic_local_subcycle_clamp=True: the wide path's "
                "per-substep clamp is LOCAL by construction (a per-substep "
                "global redistribute over the extended band would "
                "double-count the halo overlap), so the local-clamp scheme "
                "must be the EXPLICIT choice, never a silent flip.",
            )
        if config.barotropic.barotropic_diffusion_dt_ref <= 0.0:
            raise ValueError(
                f"barotropic_diffusion_dt_ref must be > 0, got "
                f"{config.barotropic.barotropic_diffusion_dt_ref!r}",
            )
        if config.min_water_column_m <= 0.0:
            raise ValueError(
                f"min_water_column_m must be > 0, got "
                f"{config.min_water_column_m!r}",
            )
        _valid_fw = {"none", "virtual_salt_flux", "real_freshwater"}
        if config.freshwater_closure == "real_freshwater":
            if (bool(getattr(config, "normalize_freshwater", False))
                    and bool(getattr(config, "barotropic_forcing_centred",
                                     False))):
                # codex RED: the centred barotropic channel averages the
                # current eta forcing with a CARRY seeded from the RAW masked
                # flux (freshwater_eta_prev), so with normalization on, the
                # centred forcing integrates to N_n/2 rather than 0 -- the
                # normalization is silently half-undone one step later.
                # Refuse until the carry itself stores the NORMALIZED field
                # (it is also seeded on the restart paths, so this is not a
                # one-line change).
                raise ValueError(
                    "freshwater_closure='real_freshwater' with "
                    "normalize_freshwater=True is not yet supported together "
                    "with barotropic_forcing_centred=True: the centred carry "
                    "(freshwater_eta_prev) stores the UN-normalized flux, so "
                    "the time-centred forcing would not integrate to zero. "
                    "Disable one of the three.")
            # codex RED: `real_freshwater` carries freshwater ONLY through the
            # eta/volume channel.  Two supported configurations discard eta
            # forcing, so freshwater would silently have NO effect at all --
            # a wrong number, not an error.  Refuse them rather than run.
            _solver = getattr(config.barotropic, "barotropic_solver", None)
            if _solver == "rigid_lid":
                raise ValueError(
                    "freshwater_closure='real_freshwater' is incompatible "
                    "with barotropic_solver='rigid_lid': the rigid-lid path "
                    "does not consume the freshwater eta forcing, and the "
                    "virtual-salt channel is deliberately absent in this "
                    "closure, so P-E+R+ice would be silently ignored. Use a "
                    "free-surface solver, or freshwater_closure="
                    "'virtual_salt_flux'.")
            if getattr(config, "prescribed_flow", None) is not None:
                raise ValueError(
                    "freshwater_closure='real_freshwater' is incompatible "
                    "with prescribed_flow: eta is reset after the barotropic "
                    "solve, discarding the freshwater volume that is this "
                    "closure's ONLY freshwater pathway.")
        # #1484 codex round-2 HIGH: the C-grid reaches the SAME shared
        # conservation fixer as MPAS, whose volume target is V_new = V_old --
        # it assumes no volume source. Under real_freshwater it would delete
        # the freshwater AFTER fix_eta_drift correctly added it, leaving the
        # whole sum(A*F)/rho_0 as residual. Guarded on MPAS in round 1; the
        # C-grid was missed, and it is reachable on either barotropic solver
        # with any explicit filter. fix_volume=False is safe for VOLUME (the
        # fixer mutates eta only inside that branch), so that stays allowed.
        if (config.freshwater_closure == "real_freshwater"
                and getattr(config, "use_conservation_fixer", False)
                and getattr(config, "fix_volume", True)):
            raise ValueError(
                'freshwater_closure="real_freshwater" is incompatible with '
                "use_conservation_fixer=True + fix_volume=True: the fixer "
                "drives V_new to V_old, which DELETES the freshwater volume "
                "source (residual = sum(A*F)/rho_0) after fix_eta_drift has "
                "added it. Set fix_volume=False, or give the fixer a "
                "freshwater-aware volume target (#1484).")
        # #1484 codex CRITICAL/HIGH: real_freshwater moves the whole
        # freshwater signal into the VOLUME channel, so any configuration that
        # does not deliver the full eta source to dV/dt silently loses mass.
        # MEASURED on the split-explicit lane with fix_eta_drift OFF: only
        # ~55% of the source reaches dV/dt (the cosine filter's average over
        # n=20 substeps), i.e. in - out - dV/dt = 0.4667*sum(A*F)/rho_0 != 0.
        # Under the OLD virtual-salt closure that volume defect was masked
        # chemically by the salt forcing; real mode removes the mask, so the
        # combination must be refused rather than run non-conserving.
        #
        # NEMO-IDENTITY EXEMPTION (decision 35, ASKED 2026-09-11).  Under
        # nemo_literal barotropic continuity the freshwater source enters eta
        # by NEMO's own substep statement (dynspg_ts.f90:553) and the
        # baroclinic step receives NEMO's filter average of it -- exactly what
        # NEMO does, and NEMO has no global projection (sshwzv.f90:137 is
        # local).  Refusing the unprojected pair there would force a
        # stabiliser the oracle lacks (Rule 9); the general lane keeps the
        # guard because its budget claim was measured on that lane.
        _nemo_literal_continuity = (
            config.barotropic.barotropic_continuity_evaluation
            == "nemo_literal")
        if (config.freshwater_closure == "real_freshwater"
                and not getattr(config, "fix_eta_drift", True)
                and not _nemo_literal_continuity):
            raise ValueError(
                'freshwater_closure="real_freshwater" requires '
                "fix_eta_drift=True: the filtered barotropic substep delivers "
                "only the filter-average of the eta source to the tracer "
                "thickness (measured ~55% at n=20 with the cosine filter), and "
                "fix_eta_drift is what projects eta onto the source-inclusive "
                "target. With it off the freshwater volume budget does not "
                "close (in - out - dV/dt != 0) and, unlike the virtual-salt "
                "closure, nothing compensates chemically (#1484). Only "
                "barotropic_continuity_evaluation='nemo_literal' is exempt: "
                "it reproduces NEMO's own unprojected budget by design.")
        if config.freshwater_closure not in _valid_fw:
            raise ValueError(
                f"freshwater_closure must be one of {_valid_fw}, "
                f"got {config.freshwater_closure!r}",
            )
        _valid_fw_sal = {"s_ref", "local"}
        if getattr(config, "freshwater_salinity", "s_ref") not in _valid_fw_sal:
            raise ValueError(
                f"freshwater_salinity must be one of {_valid_fw_sal}, "
                f"got {getattr(config, 'freshwater_salinity', 's_ref')!r}",
            )
        if (config.freshwater_closure != "real_freshwater"
                and getattr(config, "freshwater_salinity", "s_ref") == "local"
                and bool(getattr(config, "normalize_freshwater", False))):
            # Under `real_freshwater` no salinity multiplies the freshwater
            # flux at all (the VSF block is skipped), so this combination is
            # inert rather than unsound -- do not reject a legitimate config.
            # normalize_freshwater promises ZERO global salt tendency, which
            # holds only for a SCALAR S_ref (S_ref*∫F' dA = 0).  With the
            # local-S field the covariance ∫S_local·F' dA is generally
            # nonzero, silently breaking the promise (codex HIGH,
            # 2026-07-18).  Reject the combination until a joint
            # volume+salt correction exists.
            raise ValueError(
                "freshwater_salinity='local' is incompatible with "
                "normalize_freshwater=True: the zero-mean freshwater "
                "correction no longer yields zero global salt once "
                "multiplied by a spatially varying salinity "
                "(∫S_local·F' dA covariance).  Use s_ref with "
                "normalization, or local without it.")

        # Fail-fast EOS dispatch validation (dispatch discipline: validate the
        # static literal at construction, not lazily at the first step where
        # make_eos_fn would raise). Uses the single VALID_EOS_SCHEMES source so
        # the valid set is never duplicated. A linear EOS leaves eos_linear=None
        # -> make_eos_fn supplies LinearEOSConfig() defaults (documented), so no
        # eos/eos_linear coupling error is raised for that case.
        from legoesm.ocean.eos import VALID_EOS_SCHEMES
        if config.eos not in VALID_EOS_SCHEMES:
            raise ValueError(
                f"eos must be one of {sorted(VALID_EOS_SCHEMES)}, "
                f"got {config.eos!r}",
            )
        # NEMO S-EOS (``ln_seos``) coefficients are a per-run &nameos block, not
        # a library constant.  ``eos_nemo_seos=None`` keeps NemoSEOSConfig()'s
        # DINO values (every pre-existing caller).  When a card supplies its own
        # set, refuse a different EOS selection rather than silently ignoring
        # the coefficients.  GM/Redi receives the same explicit coefficient
        # object through every density, bn2 and native-slope call below.
        if getattr(config, "eos_nemo_seos", None) is not None:
            if config.eos != "nemo_seos":
                raise ValueError(
                    "eos_nemo_seos carries NEMO &nameos coefficients and is "
                    'only read when eos="nemo_seos"; got eos='
                    f"{config.eos!r}. Drop the coefficients or select the EOS.")

        # Fail-fast EKE-config validation (dispatch discipline: the EKE literals +
        # the source-augmentation flags are validated at construction). The EKE
        # SOURCE augmentation (source_kdiss_h, gm_source_mode="realized") is only
        # implemented on the 3-D (eke_3d=True) W-grid source path — reject the
        # combination with eke_3d=False rather than silently ignoring it.
        if config.gm_redi is not None and getattr(config.gm_redi, "eke", None) is not None:
            from legoesm.ocean.physics.lateral_mixing.eke import validate_eke_config
            _eke = config.gm_redi.eke
            validate_eke_config(_eke)
            if not _eke.eke_3d and (
                _eke.source_kdiss_h or _eke.gm_source_mode != "parameterized"
                or _eke.source_p_diss_iso
            ):
                raise ValueError(
                    "EKEConfig source augmentation (source_kdiss_h="
                    f"{_eke.source_kdiss_h!r}, gm_source_mode="
                    f"{_eke.gm_source_mode!r}, source_p_diss_iso="
                    f"{_eke.source_p_diss_iso!r}) requires eke_3d=True (the source "
                    "terms live on the 3-D W-grid). Set eke_3d=True or leave the "
                    "augmentation at its defaults (source_kdiss_h=False, "
                    "gm_source_mode='parameterized', source_p_diss_iso=False)."
                )

        # Fail-fast prognostic-TKE advection validation (dispatch discipline:
        # the literal is membership-checked at construction; the dispatch site
        # raises again as the factory-level tripwire). Advecting a DIAGNOSTIC
        # TKE is meaningless (no carried field) — reject the combination.
        if config.physics is not None:
            _vm = config.physics.vertical_mixing
            _tke_adv = getattr(_vm.tke, "advection_scheme", "none")
            _valid_tke_adv = {"none", "superbee"}
            if _tke_adv not in _valid_tke_adv:
                raise ValueError(
                    f"vertical_mixing.tke.advection_scheme must be one of "
                    f"{sorted(_valid_tke_adv)}, got {_tke_adv!r}")
            if _tke_adv != "none" and not (
                    _vm.scheme == "tke"
                    and bool(getattr(_vm.tke, "prognostic", False))):
                raise ValueError(
                    "vertical_mixing.tke.advection_scheme="
                    f"{_tke_adv!r} requires the PROGNOSTIC TKE closure "
                    "(vertical_mixing.scheme='tke' AND tke.prognostic=True): "
                    "only a carried TKE field can be advected (Veros "
                    "enable_tke_superbee_advection advects the prognostic "
                    f"tke[tau]). Got scheme={_vm.scheme!r}, "
                    f"prognostic={getattr(_vm.tke, 'prognostic', False)!r}.")
            if _tke_adv != "none" and not config.implicit_vertical_mixing:
                # The advective AB2 increment is applied to the TKE the
                # implicit-mixing solve returns; without that solve the
                # prognostic TKE never advances and the advection would be a
                # SILENT no-op — fail fast instead (dispatch discipline).
                raise ValueError(
                    "vertical_mixing.tke.advection_scheme="
                    f"{_tke_adv!r} requires implicit_vertical_mixing=True "
                    "(the prognostic TKE advances inside the implicit "
                    "vertical-mixing solve; without it the advection would "
                    "silently never apply).")

        # Fail-fast momentum-advection dispatch validation (was a silent
        # fallthrough to vector-invariant for any unknown literal). Single
        # source: VALID_MOMENTUM_ADVECTION in ocean_pe_latlon_cgrid.
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            VALID_MOMENTUM_ADVECTION,
            VALID_TRACER_ADVECTION,
            VALID_MOMENTUM_FLUX_SCHEME,
            VALID_VERTICAL_MOMENTUM_SCHEME,
            VALID_LATERAL_VISCOSITY_OPERATOR,
            VALID_LATERAL_VISCOSITY_E3_WEIGHTING,
            VALID_LATERAL_VISCOSITY_COEFFICIENT_SOURCE,
            VALID_CORIOLIS_SCHEME,
            VALID_AB2_SCOPE,
            VALID_WENO_SMOOTHNESS,
            VALID_LATERAL_FRICTION_SCHEME,
        )
        if config.momentum_advection not in VALID_MOMENTUM_ADVECTION:
            raise ValueError(
                f"momentum_advection must be one of "
                f"{sorted(VALID_MOMENTUM_ADVECTION)}, "
                f"got {config.momentum_advection!r}",
            )
        if config.weno_smoothness not in VALID_WENO_SMOOTHNESS:
            raise ValueError(
                f"weno_smoothness must be one of "
                f"{sorted(VALID_WENO_SMOOTHNESS)}, "
                f"got {config.weno_smoothness!r}",
            )
        # Decoupled divergence-flux smoothness: None (follow weno_smoothness) or a
        # valid family. Lets the Oceananigans recipe mix VelocityStencil vorticity
        # ("split") with OnlySelfUpwinding divergence ("standard").
        if (config.weno_divergence_smoothness is not None
                and config.weno_divergence_smoothness not in VALID_WENO_SMOOTHNESS):
            raise ValueError(
                f"weno_divergence_smoothness must be None or one of "
                f"{sorted(VALID_WENO_SMOOTHNESS)}, "
                f"got {config.weno_divergence_smoothness!r}",
            )
        if config.lateral_friction_scheme not in VALID_LATERAL_FRICTION_SCHEME:
            raise ValueError(
                f"lateral_friction_scheme must be one of "
                f"{sorted(VALID_LATERAL_FRICTION_SCHEME)}, "
                f"got {config.lateral_friction_scheme!r}",
            )
        # OM4p25 / QG-Leith are the SOLE lateral friction (Silvestri SM2 / QG2) —
        # each is ADDITIVE to the A_h/B_h/C_smag/C_leith blocks, so combining them
        # double-applies friction. Fail loudly rather than silently over-damp.
        if config.lateral_friction_scheme in ("om4p25", "qg_leith") and any(
            config.flat_get(k) > 0.0  # #501: A_h/B_h/C_smag/C_smag_lap nested in
            for k in ("A_h", "B_h", "C_smag", "C_smag_lap", "C_leith")  # lateral_viscosity
        ):
            raise ValueError(
                f"lateral_friction_scheme={config.lateral_friction_scheme!r} is the "
                "sole lateral friction but A_h/B_h/C_smag/C_smag_lap/C_leith is "
                "nonzero — these add on top and double-apply friction. Zero them in "
                "the OM4p25 (SM2) / QG-Leith (QG2) recipe.",
            )
        if config.momentum_flux_scheme not in VALID_MOMENTUM_FLUX_SCHEME:
            raise ValueError(
                f"momentum_flux_scheme must be one of "
                f"{sorted(VALID_MOMENTUM_FLUX_SCHEME)}, "
                f"got {config.momentum_flux_scheme!r}",
            )
        if config.lateral_viscosity_operator not in VALID_LATERAL_VISCOSITY_OPERATOR:
            raise ValueError(
                f"lateral_viscosity_operator must be one of "
                f"{sorted(VALID_LATERAL_VISCOSITY_OPERATOR)}, "
                f"got {config.lateral_viscosity_operator!r}",
            )
        # #1455: e3-weighting selector for the "nemo_div_curl" operator only.
        _e3w = getattr(config, "lateral_viscosity_e3_weighting", "off")
        if _e3w not in VALID_LATERAL_VISCOSITY_E3_WEIGHTING:
            raise ValueError(
                f"lateral_viscosity_e3_weighting must be one of "
                f"{sorted(VALID_LATERAL_VISCOSITY_E3_WEIGHTING)}, "
                f"got {_e3w!r}",
            )
        if _e3w != "off" and config.lateral_viscosity_operator != "nemo_div_curl":
            raise ValueError(
                "lateral_viscosity_e3_weighting != 'off' requires "
                "lateral_viscosity_operator='nemo_div_curl' (it selects the "
                f"e3-weighting of THAT operator's div/curl); got "
                f"lateral_viscosity_operator={config.lateral_viscosity_operator!r}",
            )
        # Mirrors NEMO's namdyn_ldf nn_ahm_ijk_t (ldfdyn.f90:348-353 is the
        # -30 arm); same guard as the e3-weighting selector above.
        _ahm_src = getattr(
            config, "lateral_viscosity_coefficient_source", "nemo_ldf_c2d")
        if _ahm_src not in VALID_LATERAL_VISCOSITY_COEFFICIENT_SOURCE:
            raise ValueError(
                f"lateral_viscosity_coefficient_source must be one of "
                f"{sorted(VALID_LATERAL_VISCOSITY_COEFFICIENT_SOURCE)}, "
                f"got {_ahm_src!r}",
            )
        if (_ahm_src != "nemo_ldf_c2d"
                and config.lateral_viscosity_operator != "nemo_div_curl"):
            raise ValueError(
                "lateral_viscosity_coefficient_source != 'nemo_ldf_c2d' "
                "requires lateral_viscosity_operator='nemo_div_curl' (it "
                "selects where THAT operator's ahmt/ahmf come from); got "
                f"lateral_viscosity_operator={config.lateral_viscosity_operator!r}",
            )
        _vert_mom_scheme = getattr(
            config, "vertical_momentum_scheme", "upwind_perturbation")
        if _vert_mom_scheme not in VALID_VERTICAL_MOMENTUM_SCHEME:
            raise ValueError(
                f"vertical_momentum_scheme must be one of "
                f"{sorted(VALID_VERTICAL_MOMENTUM_SCHEME)}, "
                f"got {_vert_mom_scheme!r}",
            )
        if (_vert_mom_scheme == "nemo_up3"
                and (config.momentum_advection != "flux_form"
                     or config.momentum_flux_scheme != "nemo_up3")):
            raise ValueError(
                "vertical_momentum_scheme='nemo_up3' is one dynadv_up3 "
                "program and requires momentum_advection='flux_form' with "
                "momentum_flux_scheme='nemo_up3' (the NEMO-referenced "
                "horizontal arm; 'oceananigans_up3' is a different reference)")
        # #1226 level-29-onset fix: bottom/straddling-face mask convention for
        # nemo_advective_vertical_momentum_advection (only consumed under
        # vertical_momentum_scheme="nemo_advective"; validated unconditionally
        # here so a typo on ANY recipe fails fast at config construction).
        from legoesm.ocean.vertical import VALID_ZAD_BOTTOM_FACE_MASK
        _zad_mask_mode = getattr(config, "zad_bottom_face_mask", "min_rule")
        if _zad_mask_mode not in VALID_ZAD_BOTTOM_FACE_MASK:
            raise ValueError(
                f"zad_bottom_face_mask must be one of "
                f"{sorted(VALID_ZAD_BOTTOM_FACE_MASK)}, "
                f"got {_zad_mask_mode!r}",
            )
        _zad_qco = getattr(config, "zad_qco_evaluation", "generic")
        if _zad_qco not in {"generic", "nemo_literal"}:
            raise ValueError(
                "zad_qco_evaluation must be one of ['generic', "
                f"'nemo_literal'], got {_zad_qco!r}")
        if _zad_qco == "nemo_literal" and _vert_mom_scheme != "nemo_advective":
            raise ValueError(
                "zad_qco_evaluation='nemo_literal' requires "
                "vertical_momentum_scheme='nemo_advective'; the QCO ww and "
                "live Kmm thickness operands are a coupled dynzad path")
        _wzv_call2 = getattr(config, "wzv_call2_evaluation", "generic")
        if _wzv_call2 not in {"generic", "nemo_literal"}:
            raise ValueError(
                "wzv_call2_evaluation must be one of ['generic', "
                f"'nemo_literal'], got {_wzv_call2!r}")
        if _wzv_call2 == "nemo_literal" and _zad_qco != "nemo_literal":
            raise ValueError(
                "wzv_call2_evaluation='nemo_literal' requires "
                "zad_qco_evaluation='nemo_literal'; second hdiv and "
                "barotropic Kaa r3t are an inseparable QCO operand pair")
        # Reject centered_full / nemo_advective + adaptive-implicit vertadv:
        # the adaptive-implicit path (ln_zad_Aimp) replaces the explicit
        # in-tendency vertical momentum advection ENTIRELY with an upwind
        # backward-Euler solve at the step level, so an explicit-stage
        # option here would be a silent no-op.  Fail fast rather than mislead.
        if (_vert_mom_scheme in ("centered_full", "nemo_advective")
                and getattr(config, "adaptive_implicit_vertadv", False)):
            raise ValueError(
                f"vertical_momentum_scheme={_vert_mom_scheme!r} is "
                "incompatible with adaptive_implicit_vertadv=True: the "
                "adaptive-implicit scheme replaces the explicit vertical "
                "momentum advection entirely (upwind backward-Euler at the "
                "step level), so the explicit tendency would never be "
                "applied. Choose one.",
            )

        # The lat-lon C-grid applies OceanSurfaceForcing.tau/q_net/salt DIRECTLY
        # in its dynamics (+ the freshwater= arg for eta + virtual salt), so it
        # must NOT also route the same forcing through the 'external' physics
        # scheme — that would double-apply momentum/heat/salt.  'external' is the
        # cubed-sphere physics-path coupling route; lat-lon uses the direct path.
        if (config.physics is not None
                and config.physics.surface_forcing.scheme == "external"):
            raise ValueError(
                "surface_forcing.scheme='external' is not supported on the "
                "lat-lon C-grid ocean: it applies OceanSurfaceForcing directly "
                "in its dynamics, so the external physics scheme would "
                "double-apply the forcing. Pass surface_forcing= (and "
                "freshwater=) to step() instead.",
            )
        if config.runtime_checks.max_abs_eta_m <= 0.0:
            raise ValueError(
                f"max_abs_eta_m must be > 0, got {config.runtime_checks.max_abs_eta_m!r}")
        if config.runtime_checks.temperature_min_c >= config.runtime_checks.temperature_max_c:
            raise ValueError(
                f"temperature_min_c ({config.runtime_checks.temperature_min_c}) must be "
                f"< temperature_max_c ({config.runtime_checks.temperature_max_c})")
        if config.runtime_checks.salinity_min_psu >= config.runtime_checks.salinity_max_psu:
            raise ValueError(
                f"salinity_min_psu ({config.runtime_checks.salinity_min_psu}) must be "
                f"< salinity_max_psu ({config.runtime_checks.salinity_max_psu})")
        _valid_solvers = {"explicit_substep", "implicit_cn", "rigid_lid",
                          "implicit_unsplit"}
        if config.barotropic.barotropic_solver not in _valid_solvers:
            raise ValueError(
                f"barotropic_solver must be one of {_valid_solvers}, "
                f"got {config.barotropic.barotropic_solver!r}")
        if config.barotropic.barotropic_solver == "implicit_unsplit":
            # The unsplit step (_unsplit_ab2_step) integrates self.tendencies()
            # (the baroclinic tendencies) + one implicit free-surface solve +
            # _apply_implicit_vertical_mixing.  It does NOT yet thread the extra
            # physics the split _step_impl/_ab2_step path carries, so REJECT configs
            # that use them rather than silently dropping them (CLAUDE.md dispatch-
            # hardening / "no silent coerce").  The freshwater step-arg is gated in
            # _unsplit_ab2_step itself.
            _unsupported = []
            # Standard GM/Redi (isopycnal + skew tracer mixing, incl. implicit_K33)
            # IS supported by _unsplit_ab2_step; only the prognostic-EKE-coupled GM
            # path (gm_redi.eke) is not yet threaded.
            if (getattr(config, "gm_redi", None) is not None
                    and getattr(config.gm_redi, "eke", None) is not None):
                _unsupported.append("gm_redi with prognostic EKE (gm_redi.eke)")
            # gm_bolus_advection="through_fct" needs the bolus-transport export
            # + add_bolus_to_advecting_flux wiring, which only the split step
            # carries; without this gate the unsplit step would silently DROP
            # the entire GM bolus term (the in-operator centred add is gated
            # off and nothing re-adds it to the advecting flux).
            if (getattr(config, "gm_redi", None) is not None
                    and getattr(config.gm_redi, "gm_bolus_advection",
                                "centred") == "through_fct"):
                _unsupported.append(
                    'gm_redi gm_bolus_advection="through_fct" (bolus-through-'
                    "FCT flux not threaded by the unsplit step)")
            if getattr(config, "ab2_scope", "total") == "advective":
                _unsupported.append('ab2_scope="advective" (withheld dissipation)')
            if getattr(config, "surface_forcing_implicit", False):
                _unsupported.append("surface_forcing_implicit")
            if getattr(config, "sponge_forcing_implicit", False):
                _unsupported.append("sponge_forcing_implicit")
            if getattr(config, "momentum_friction_additive", False):
                _unsupported.append("momentum_friction_additive")
            _vm = getattr(getattr(config, "physics", None), "vertical_mixing", None)
            if (_vm is not None and getattr(_vm, "scheme", None) == "tke"
                    and getattr(getattr(_vm, "tke", None), "prognostic", False)):
                _unsupported.append("prognostic TKE")
            # store_mass_flux (#1442, codex RED 5): the capture lives in
            # _step_impl, which _unsplit_ab2_step BYPASSES entirely (it advects
            # tracers through the advective-form tendency -- there is no shared
            # mass-flux block to capture).  Silently, the slots would stay None
            # on step 1 and then hold a STALE value forever once seeded, which a
            # transport diagnostic would integrate as if it were live.  Reject,
            # exactly as prescribed_flow is rejected for the same structural
            # reason (see _validate_config's prescribed_flow guard).
            if getattr(config, "store_mass_flux", False):
                _unsupported.append(
                    "store_mass_flux (the tracer-advecting flux is formed "
                    "inside the advective-form tendency, not in a shared "
                    "mass-flux block the step can capture)")
            if getattr(config, "store_salt_flux", False):
                # codex round-1 RED 3: same bypass -- _unsplit_ab2_step never
                # runs _step_impl's capture, so the seeded 2-D slots would sit
                # at ZERO forever and require_salt=True would happily integrate
                # an exact salt transport of 0 (a lie with the right shape).
                _unsupported.append(
                    "store_salt_flux (the advective salt flux is formed "
                    "inside the advective-form tendency the unsplit step "
                    "uses; the capture lives in _step_impl)")
            if _unsupported:
                raise ValueError(
                    'barotropic_solver="implicit_unsplit" does not yet support: '
                    + ", ".join(_unsupported)
                    + ". The unsplit free-surface step carries only the baroclinic "
                    "tendencies + implicit FS + implicit vertical mixing; these "
                    "features are threaded by the split implicit_cn path only. Use "
                    'barotropic_solver="implicit_cn", or extend _unsplit_ab2_step.')
        _valid_time_filters = {"box", "cosine", "power_law",
                               "nemo_boxcar_centred", "nemo_ab3am4",
                               "nemo_boxcar_ab3", "nemo_boxcar1_ab3"}
        if (getattr(config, "surface_stress_implicit", False)
                and not getattr(config.barotropic,
                                "nemo_stage_mean_imposition", False)
                and config.barotropic.barotropic_solver != "explicit_substep"):
            # Under the split-explicit free surface ("explicit_substep") the
            # post-solve depth-mean shift feeds the next step's Kbb barotropic
            # seed, and F_slow carries the wind for the substeps exactly as
            # NEMO's zu_frc wind term (dynspg_ts.F90 ~L360).  So no stage-mean
            # imposition is required there; other solvers keep the guard.
            #
            # CORRECTED 2026-08-21 (#1455): this comment used to justify the
            # exemption by claiming NEMO's MLF has "no post-zdf re-imposition".
            # THAT IS FALSE -- stpmlf.F90:578 calls mlf_baro_corr AFTER dyn_zdf
            # (:396) and it re-imposes the barotropic mean on the committed
            # after level. The exemption still stands on its OTHER leg (F_slow
            # already carries the wind into the substeps here), which is the
            # one that was doing the work; the false half is removed rather
            # than left to be cited. legoESM can now run NEMO's post-zdf site
            # explicitly -- see BarotropicConfig.barotropic_after_reconcile.
            raise ValueError(
                "surface_stress_implicit=True deposits the wind stress inside "
                "the implicit vertical solve, which SHIFTS the depth mean "
                "after the barotropic solve; outside "
                'barotropic_solver="explicit_substep" (NEMO stpmlf order) it '
                "requires barotropic.nemo_stage_mean_imposition=True (NEMO "
                "stprk3_stg:440) to re-impose the barotropic mean — "
                "otherwise the wind's depth-mean is double-counted "
                "(F_slow + the solve).")
        if (config.barotropic.barotropic_time_filter
                in ("nemo_ab3am4", "nemo_boxcar_ab3", "nemo_boxcar1_ab3")
                and config.barotropic.barotropic_wide_halo):
            raise ValueError(
                f"barotropic_time_filter="
                f"{config.barotropic.barotropic_time_filter!r} is not yet "
                "supported with barotropic_wide_halo=True (the AB3/AM4 substep "
                "histories widen the per-substep stencil reach; the wide-halo "
                "budget has not been re-derived).")
        if config.barotropic.barotropic_time_filter not in _valid_time_filters:
            raise ValueError(
                f"barotropic_time_filter must be one of {_valid_time_filters}, "
                f"got {config.barotropic.barotropic_time_filter!r}")
        # nemo_boxcar_ab3 (NEMO nn_bt_flt=2) is only flt=2-faithful under the
        # MLF leap-frog family (_leapfrog_step OR nemo_mlf's _nemo_mlf_step --
        # both supply the SAME ×2 substep scale + Nbb before-level seed, per
        # _nemo_mlf_step's docstring "barotropic-before seeding ... IDENTICAL
        # to _leapfrog_step"), which make the boxcar 2*nn_e-wide and centred at
        # Naa (dynspg_ts.F90:1061, 494-503). Pairing it with forward_euler
        # would give a flt=1-width window + AB3 + ts_bck_interp — a non-NEMO
        # hybrid — so reject it (only the leapfrog-family cards select it).
        if (config.barotropic.barotropic_time_filter == "nemo_boxcar_ab3"
                and getattr(config, "outer_integrator", "forward_euler")
                not in ("leapfrog", "nemo_mlf")):
            raise ValueError(
                'barotropic_time_filter="nemo_boxcar_ab3" (NEMO nn_bt_flt=2 AB3 '
                "+ ts_bck_interp dissipation) requires outer_integrator in "
                '("leapfrog", "nemo_mlf") (the MLF family supplies the ×2 '
                "substep scale + before-level seed that make the boxcar the "
                "faithful 2*nn_e window centred at Naa); got outer_integrator="
                f"{getattr(config, 'outer_integrator', 'forward_euler')!r}. Use "
                'barotropic_time_filter="nemo_boxcar_centred" for the '
                "forward-frame boxcar.")
        # zdf_drag_in_matrix (#1226 dynzdf.F90:293-305): the diagonal term is
        # NEMO's rCdU_bot (zdfdrg zdf_drg_nonlin/loglayer), so it requires a
        # NEMO bottom-drag scheme (the legacy linear / MOM6 DRAG_BG_VEL rate
        # is not what NEMO's ln_drgimp path uses) AND the implicit vmix solve
        # itself (there is no matrix to add the diagonal term into otherwise).
        if getattr(config, "zdf_drag_in_matrix", False):
            if not getattr(config, "implicit_vertical_mixing", False):
                raise ValueError(
                    "zdf_drag_in_matrix=True requires "
                    "implicit_vertical_mixing=True (NEMO ln_drgimp adds the "
                    "drag into the implicit vertical-friction tridiagonal "
                    "matrix; there is no matrix without the implicit solve).")
            _bd_scheme = getattr(config.bottom_drag, "bottom_drag_scheme",
                                  "legacy")
            if _bd_scheme not in ("nemo_quadratic", "nemo_loglayer",
                                  "nemo_linear"):
                raise ValueError(
                    "zdf_drag_in_matrix=True requires bottom_drag_scheme in "
                    '{"nemo_quadratic", "nemo_loglayer", "nemo_linear"} '
                    "(NEMO's zdfdrg rCdU_bot rate, whichever of ln_non_lin / "
                    "ln_loglayer / ln_lin the run selects); got "
                    f"{_bd_scheme!r}.")
            # zdf_drag_in_matrix skips the explicit _bc_bottom_drag RHS kick
            # (single-owner guard, ocean_pe_latlon_cgrid.py) to avoid double-
            # counting drag in the 3-D momentum tendency. But under
            # barotropic_solver="explicit_substep", _bc_bottom_drag is ALSO
            # the barotropic substep's ONLY default drag source in F_slow
            # (barotropic_latlon_cgrid.py substep-loop drag comment) — the
            # implicit-matrix diagonal built here never reaches the
            # barotropic mode. NEMO covers the split-explicit barotropic
            # with its OWN in-subcycle drag under ln_dynspg_ts (dyn_drg,
            # dynspg_ts.F90:700-706 + dyn_drg_init :1584-1642), transcribed
            # as barotropic_drag_substep — required here so the barotropic
            # mode is never silently undamped.
            if (config.barotropic.barotropic_solver == "explicit_substep"
                    and not getattr(config, "barotropic_drag_substep",
                                    False)):
                raise ValueError(
                    "zdf_drag_in_matrix=True with "
                    'barotropic_solver="explicit_substep" requires '
                    "barotropic_drag_substep=True: zdf_drag_in_matrix skips "
                    "_bc_bottom_drag's RHS kick (single-owner guard, no "
                    "double-count in the 3-D matrix), which is the "
                    "explicit_substep barotropic loop's ONLY default drag "
                    "source in F_slow — without the in-subcycle substep "
                    "drag (NEMO dyn_drg, dynspg_ts.F90:700-706 + 1584-1642) "
                    "the barotropic mode would run completely undamped. "
                    "Enable barotropic_drag_substep=True (NEMO's DINO "
                    "composition), use a different barotropic_solver, or "
                    "leave zdf_drag_in_matrix=False.")
        _zdf_solver_evaluation = getattr(
            config, "zdf_implicit_solver_evaluation", "shared_thomas")
        if _zdf_solver_evaluation not in ("shared_thomas", "nemo_literal"):
            raise ValueError(
                "unknown zdf_implicit_solver_evaluation "
                f"{_zdf_solver_evaluation!r}; expected 'shared_thomas' or "
                "'nemo_literal'")
        # barotropic_drag_substep (#1226, NEMO dyn_drg): the in-subcycle
        # explicit barotropic drag + pu_RHSi slow-forcing correction.
        if getattr(config, "barotropic_drag_substep", False):
            if not getattr(config, "zdf_drag_in_matrix", False):
                raise ValueError(
                    "barotropic_drag_substep=True requires "
                    "zdf_drag_in_matrix=True: with the default explicit 3-D "
                    "drag kick (_bc_bottom_drag) active, F_slow already "
                    "carries the depth-mean of drag on the FULL bottom "
                    "velocity into every substep — adding the in-subcycle "
                    "substep drag + the dyn_drg_init pu_RHSi correction on "
                    "top would double-count the barotropic drag. NEMO's "
                    "DINO composition (ln_drgimp=T + ln_dynspg_ts=T) maps "
                    "to zdf_drag_in_matrix=True + zdf_baroclinic_only=True "
                    "+ barotropic_drag_substep=True.")
            # NEMO dynzdf.F90:147-159 UNCONDITIONALLY (under ln_drgimp +
            # ln_dynspg_ts) removes the barotropic mean uu_b/vv_b from the
            # 3-D implicit solve, so the in-matrix drag diagonal acts on the
            # baroclinic residual only and the barotropic mode is dragged
            # ONCE — by dynspg_ts's in-subcycle drag.  In lego that removal
            # is the zdf_baroclinic_only / nemo_stage_mean_imposition
            # machinery; without it the extra_diag drags the FULL bottom
            # velocity (barotropic included, surviving into the 3-D depth
            # mean) AND the substep drag damps the barotropic mode again —
            # a double count (adversarial-review finding 1).
            if not (getattr(config, "zdf_baroclinic_only", False)
                    or getattr(config.barotropic,
                               "nemo_stage_mean_imposition", False)):
                raise ValueError(
                    "barotropic_drag_substep=True requires "
                    "zdf_baroclinic_only=True (or "
                    "barotropic.nemo_stage_mean_imposition=True): NEMO "
                    "removes the barotropic mean from the 3-D implicit "
                    "solve unconditionally under ln_drgimp + ln_dynspg_ts "
                    "(dynzdf.F90:147-159), so the in-matrix drag acts on "
                    "the baroclinic residual only. Without that removal "
                    "the matrix diagonal drags the FULL bottom velocity "
                    "(barotropic included) and the in-subcycle substep "
                    "drag double-counts the barotropic mode.")
            if config.barotropic.barotropic_solver != "explicit_substep":
                raise ValueError(
                    "barotropic_drag_substep=True is the explicit_substep "
                    "in-subcycle drag (NEMO dyn_drg, dynspg_ts.F90:700-706); "
                    "under barotropic_solver="
                    f"{config.barotropic.barotropic_solver!r} it would be a "
                    "partial mechanism (only the F_slow pu_RHSi correction "
                    "would apply). Use "
                    'barotropic_solver="explicit_substep" or leave the flag '
                    "False.")
        # Loud no-op guard: barotropic_time_filter is consumed ONLY by the split-
        # explicit substep (barotropic_substeps_latlon_cgrid). The implicit_cn /
        # implicit_unsplit / rigid_lid solvers have no barotropic substep to filter
        # and silently ignore it — so a non-default filter under an implicit solver
        # reads as "applied" while doing nothing. Warn (not raise: harmless, just
        # inert) so the silent no-op is visible.
        if (config.barotropic.barotropic_time_filter != "cosine"
                and config.barotropic.barotropic_solver != "explicit_substep"):
            import warnings
            warnings.warn(
                f"barotropic_time_filter={config.barotropic.barotropic_time_filter!r} "
                f"has NO effect under barotropic_solver="
                f"{config.barotropic.barotropic_solver!r}: the time filter is consumed "
                "only by the explicit_substep barotropic substep. Use "
                'barotropic_solver="explicit_substep" to apply it, or leave the filter '
                'at its "cosine" default to silence this warning.',
                stacklevel=3)
        # Single source: VALID_TRACER_ADVECTION mirrors the flux-form tendency
        # dispatch (its else-raise) plus the SOM special case handled in step().
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            VALID_TRACER_ADVECTION as _valid_tracer_adv,
        )
        if config.tracer_advection not in _valid_tracer_adv:
            raise ValueError(
                f"tracer_advection must be one of {sorted(_valid_tracer_adv)}, "
                f"got {config.tracer_advection!r}")
        _valid_outer_int = {"forward_euler", "ab2", "leapfrog", "nemo_mlf"}
        _outer_int = getattr(config, "outer_integrator", "forward_euler")
        if _outer_int not in _valid_outer_int:
            raise ValueError(
                f"outer_integrator must be one of {sorted(_valid_outer_int)}, "
                f"got {_outer_int!r}")
        # The two outer steps that have NEMO's stpmlf.F90 stage layout. Defined
        # here because the first consumer is the guard immediately below.
        _leapfrog_family = ("leapfrog", "nemo_mlf")
        # barotropic_after_reconcile (NEMO mlf_baro_corr) has a site ONLY in
        # the two leap-frog-family outer steps. Selecting it on forward_euler
        # or ab2 would silently do NOTHING -- a card asking for a
        # reconciliation and not getting one, which is exactly the failure the
        # fn-entry dispatch guard exists to prevent and which no typo is needed
        # to reach (forward_euler is the DEFAULT outer_integrator). Reject at
        # construction instead.
        _after_recon = getattr(
            config.barotropic, "barotropic_after_reconcile", "off")
        if _after_recon != "off" and _outer_int not in _leapfrog_family:
            raise ValueError(
                f"barotropic_after_reconcile={_after_recon!r} requires "
                f"outer_integrator in {sorted(_leapfrog_family)}: NEMO's "
                "mlf_baro_corr (stpmlf.F90:754-765) runs after dyn_zdf and "
                "before the Asselin filter, and only the leap-frog-family "
                "outer steps have that position. Got "
                f"outer_integrator={_outer_int!r}, which would silently ignore "
                "the setting.")
        # "leapfrog" (_leapfrog_step, two-pass) and "nemo_mlf" (_nemo_mlf_step,
        # P2 single-pass transcription, docs/ocean/fidelity/
        # nemo_mlf_step_transcription_spec.md) are the SAME leap-frog-family
        # composition (rdt=2dt, Asselin filter, _ab2_scope_override="advective",
        # centred-forcing carry, barotropic-before seed -- see _nemo_mlf_step's
        # docstring: "IDENTICAL to _leapfrog_step" for everything except the
        # dissipative-term composition mechanism). Every guard below was audited
        # PER GUARD (spec §4): none of the leapfrog-specific requirements are
        # moot for nemo_mlf -- _nemo_mlf_step reads config.ab2_scope through the
        # SAME collision path (still always passes _ab2_scope_override=
        # "advective"), calls the SAME _seed_centred_forcing_carry, and has no
        # prescribed_flow re-pinning either -- so every one of these raises is
        # extended to the pair, not blanket-copied without re-checking.
        # (_leapfrog_family is defined above, at its first consumer.)
        # NEMO Modified-Leap-Frog (stpmlf.F90) requirements. The leapfrog carries
        # the Coriolis in the explicit RHS (via vorticity_scheme="een_total") and
        # applies the vertical friction/mixing implicitly over 2dt (dyn_zdf), so
        # it needs coriolis_scheme="explicit_ab2" (Matsuno off) + implicit vertical
        # mixing. Reject silent misconfiguration rather than run a non-NEMO scheme.
        if _outer_int in _leapfrog_family:
            _asg = getattr(config, "asselin_gamma", 0.1)
            if not (0.0 <= _asg <= 1.0):
                raise ValueError(
                    "asselin_gamma (Robert-Asselin filter coefficient rn_atfp) "
                    f"must be in [0, 1]; got {_asg!r}")
            if getattr(config, "coriolis_scheme", "matsuno_split") != "explicit_ab2":
                raise ValueError(
                    f'outer_integrator={_outer_int!r} (NEMO stp_MLF) requires '
                    'coriolis_scheme="explicit_ab2": the leapfrog carries the '
                    "Coriolis in the explicit RHS (Matsuno rotation OFF). Use it "
                    'with vorticity_scheme="een_total" (NEMO ln_dynvor_een, f in '
                    "the EEN triad). Got coriolis_scheme="
                    f'{getattr(config, "coriolis_scheme", "matsuno_split")!r}.')
            if not config.implicit_vertical_mixing:
                raise ValueError(
                    f'outer_integrator={_outer_int!r} requires '
                    "implicit_vertical_mixing=True: NEMO dyn_zdf applies the "
                    "vertical friction/diffusion as a backward-Euler solve over "
                    "rDt=2dt on the leapfrog after-state.")
            if getattr(config, "ab2_scope", "total") != "total":
                raise ValueError(
                    f'outer_integrator={_outer_int!r} requires ab2_scope="total" '
                    "(the leapfrog-family step applies every explicit tendency "
                    "at the same 2dt weight via its OWN internal "
                    '_ab2_scope_override="advective" -- the config-level '
                    "advective/dissipative split is AB2-only).")
            if getattr(config, "prescribed_flow", None) is not None:
                raise ValueError(
                    f'outer_integrator={_outer_int!r} does not support the '
                    "prescribed_flow lever (neither _leapfrog_step nor "
                    "_nemo_mlf_step re-pins the flow after the barotropic/"
                    "implicit solves, unlike _step_impl/_ab2_step). Use "
                    "forward_euler or ab2 with prescribed_flow.")
        if _outer_int == "nemo_mlf":
            # Resolved decision 4 (spec §6-4/§7 P2): a transcription that still
            # permits a non-NEMO implicit-solve divisor stops being a
            # transcription at that row (stpmlf.F90 row 22/29, trazdf.F90:
            # 219-220 e3w(Kmm)) -- the standalone-A/B NULL result
            # (DINO_NEMO_KMM_DIVISOR measured no climate effect on the EXISTING
            # leapfrog card) governs only whether to flip that card's default;
            # it does not transfer to nemo_mlf, which changes the surrounding
            # composition the divisor sits inside.
            if getattr(config, "zdf_implicit_solver_evaluation",
                       "shared_thomas") != "nemo_literal":
                raise ValueError(
                    'outer_integrator="nemo_mlf" requires '
                    'zdf_implicit_solver_evaluation="nemo_literal": nemo_mlf '
                    "is a literal transcription of stpmlf.F90's dyn_zdf/tra_zdf "
                    "calls, whose implicit-solve gradient divisor is e3w(Kmm) "
                    "(trazdf.F90:219-221), not legoESM's default after-solve "
                    "midpoint slot. That divisor is part of the NEMO identity, "
                    'so select it: zdf_implicit_solver_evaluation="nemo_literal".')
            # RETRACTED 2026-08-21 (#1455): the waiver this guard used to
            # rest on said mlf_baro_corr (stpmlf.F90 row 30) was "PROVABLY a
            # no-op" under surface_stress_implicit=False, because no depth-mean
            # source was thought to exist. THAT IS REFUTED by measurement:
            # `barotropic_after_reconcile="nemo_mlf_baro_corr"` shifts
            # _nemo_mlf_step's after-state by 2.0e-4 m/s on a partial-cell
            # fixture with surface_stress_implicit=False
            # (tests/ocean/unit/test_barotropic_after_reconcile.py), and on the
            # 90-day DINO twin the discarded column mean is +0.27 m3/s2 per
            # southern u-row. The implicit vertical solve IS a depth-mean
            # source, through the weighting difference between its own strip/
            # re-add and the combine's split.
            #
            # The kernel the old message told the reader to "build first" NOW
            # EXISTS (barotropic_common.after_level_column_mean_reconcile) and
            # is wired into BOTH outer-step paths via
            # _apply_after_level_reconcile, so nemo_mlf is no longer waiving
            # anything -- it runs row 30 whenever the card selects it. What
            # remains true is the narrower statement below: legoESM has no
            # implicit surface-stress source on this path, so that combination
            # is still untranscribed and still rejected.
            if getattr(config, "surface_stress_implicit", False):
                raise ValueError(
                    'outer_integrator="nemo_mlf" does not support '
                    "surface_stress_implicit=True: this path does not "
                    "transcribe NEMO's implicit surface-stress boundary "
                    "condition, so the after-level state would carry a "
                    "depth-mean source NEMO builds differently. Note this is "
                    "no longer a statement about mlf_baro_corr, which IS now "
                    'available on this path via barotropic_after_reconcile='
                    '"nemo_mlf_baro_corr".')
        if (getattr(config, "barotropic_forcing_centred", False)
                and _outer_int not in _leapfrog_family):
            raise ValueError(
                'barotropic_forcing_centred=True requires outer_integrator in '
                '("leapfrog", "nemo_mlf"): the ½(before+now) forcing '
                "average (NEMO ln_bt_fw=.FALSE., dynspg_ts.F90:392-421) and "
                "the drag-residual BEFORE level (:1623-1636) both read the "
                "state's u_before/v_before/tau_x_prev/tau_y_prev/"
                "freshwater_eta_prev carry fields, which only exist under "
                "the leapfrog-family (NEMO Modified-Leap-Frog) time integrator. "
                f"Got outer_integrator={_outer_int!r}.")
        # TKE closure axes that read the leap-frog-family BEFORE (Nbb) state
        # (T4 Burchard shear, T8/T13 rn2b Prandtl/Langmuir; Phase-2 #1317):
        # both need state.u_before/v_before/T_before/S_before, which exist
        # under EITHER outer_integrator="leapfrog" OR "nemo_mlf" (same Nbb
        # carry mechanism -- _nemo_mlf_step's Asselin-filter tail populates
        # them identically to _leapfrog_step's, per its docstring). Not moot
        # for nemo_mlf: audited, not blanket-copied. Construction-time raise
        # (dispatch hardening) rather than a silent no-op at model-step time.
        _vmix_cfg_ctor = getattr(getattr(config, "physics", None),
                                 "vertical_mixing", None)
        if _vmix_cfg_ctor is not None and _vmix_cfg_ctor.scheme == "tke":
            _tke_cfg_ctor = _vmix_cfg_ctor.tke
            if (getattr(_tke_cfg_ctor, "tke_n2_time_level", "step_entry")
                    == "nemo_before" and _outer_int not in _leapfrog_family
                    and getattr(config, "momentum_time_integrator", "euler")
                    != "rk3_ws"):
                raise ValueError(
                    'vertical_mixing.tke.tke_n2_time_level="nemo_before" '
                    'requires outer_integrator in ("leapfrog", "nemo_mlf") '
                    'or momentum_time_integrator="rk3_ws": '
                    "the true rn2b (Nbb) tracers only exist as "
                    "state.T_before/S_before under the leap-frog-family "
                    "integrator, while NEMO RK3 defines Nbb as the whole-step "
                    f"entry state. Got "
                    f"outer_integrator={_outer_int!r}.")
            _tke_shear_ctor = getattr(_tke_cfg_ctor, "tke_shear_production",
                                     "squared_centered")
            # "nemo_face_native_nbb2" = face-native SPATIAL geometry with both
            # operands at the Nbb whole-step-entry slot.  key_RK3's live call
            # is zdf_phy(kstp,Nbb,Nbb,Nrhs), stprk3.F90:164-165; the commented
            # Nbb,Nnn form is MLF semantics, not this arm.
            if (_tke_shear_ctor in ("nemo_burchard", "nemo_face_native")
                    and _outer_int not in _leapfrog_family):
                raise ValueError(
                    f'vertical_mixing.tke.tke_shear_production='
                    f'{_tke_shear_ctor!r} requires outer_integrator in '
                    '("leapfrog", "nemo_mlf"): both the Burchard now×before '
                    "cross term and its face-native extension only exist as "
                    "state.u_before/v_before under the leap-frog-family (NEMO "
                    f"Modified-Leap-Frog) time integrator. Got "
                    f"outer_integrator={_outer_int!r}.")
        # Distributed fixed-iteration PCG knobs (implicit_cn under MPI).
        if config.barotropic.barotropic_implicit_pcg_fixed_iters < 1:
            raise ValueError(
                "barotropic_implicit_pcg_fixed_iters must be >= 1 "
                "(the distributed PCG runs exactly this many iterations); "
                f"got {config.barotropic.barotropic_implicit_pcg_fixed_iters!r}")
        if config.barotropic.barotropic_implicit_pcg_residual_tol <= 0.0:
            raise ValueError(
                "barotropic_implicit_pcg_residual_tol must be > 0 "
                f"(diagnostic acceptance tol); got "
                f"{config.barotropic.barotropic_implicit_pcg_residual_tol!r}")
        # Asynchronous dt_mom≠dt_tracer stepping (dt_mom = dt / dt_mom_ratio).
        if config.dt_mom_ratio < 1.0:
            raise ValueError(
                "dt_mom_ratio must be >= 1.0 (dt_mom <= dt_tracer), "
                f"got {config.dt_mom_ratio!r}")
        if config.dt_mom_ratio != 1.0 and config.barotropic.barotropic_solver != "rigid_lid":
            raise ValueError(
                "dt_mom_ratio != 1.0 (asynchronous dt_mom≠dt_tracer stepping) "
                "requires barotropic_solver='rigid_lid': under the rigid lid the "
                "column depth H is fixed, so the tracer flux-form update is exactly "
                "dt-independent and tracer mass is conserved (matching Veros's "
                "streamfunction rigid lid). A moving free surface would leak "
                "O((dt_tracer-dt_mom)·∂h/∂t) tracer mass. Got barotropic_solver="
                f"{config.barotropic.barotropic_solver!r}.")
        if config.barotropic.rigid_lid_cg_tol <= 0.0:
            raise ValueError(
                f"rigid_lid_cg_tol must be > 0, got {config.barotropic.rigid_lid_cg_tol!r}")
        if config.barotropic.rigid_lid_cg_maxiter < 1:
            raise ValueError(
                f"rigid_lid_cg_maxiter must be >= 1, "
                f"got {config.barotropic.rigid_lid_cg_maxiter!r}")
        for fld in ("barotropic_implicit_theta_eta",
                    "barotropic_implicit_theta_pgf"):
            v = getattr(config.barotropic, fld)  # #501: nested BarotropicConfig
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{fld} must be in [0, 1], got {v!r}")
        if config.barotropic.barotropic_implicit_pcg_tol <= 0.0:
            raise ValueError(
                f"barotropic_implicit_pcg_tol must be > 0, "
                f"got {config.barotropic.barotropic_implicit_pcg_tol!r}")
        if config.barotropic.barotropic_implicit_pcg_maxiter < 1:
            raise ValueError(
                f"barotropic_implicit_pcg_maxiter must be >= 1, "
                f"got {config.barotropic.barotropic_implicit_pcg_maxiter!r}")
        _valid_pgf = {"adcroft", "smc03", "nemo_sco"}
        pgf_scheme = getattr(config, "pgf_scheme", "adcroft")
        if pgf_scheme not in _valid_pgf:
            raise ValueError(
                f"pgf_scheme must be one of {_valid_pgf}, got {pgf_scheme!r}")
        if (pgf_scheme == "nemo_sco"
                and getattr(config, "pgf_quadrature", "cell_integral")
                != "nemo_trapezoid"):
            # NEMO hpg_sco's zuap/stretch terms telescope against the dynhpg
            # trapezoid p' on the SAME gdept ladder; pairing them with the
            # midpoint cell_integral rule would leave a spurious rest-η PGF at
            # staircase steps (fn-entry guard duplicated in
            # _bc_ke_and_pressure_gradients for direct callers).
            raise ValueError(
                'pgf_scheme="nemo_sco" requires pgf_quadrature='
                '"nemo_trapezoid" (NEMO dynhpg pairs the hpg_sco stencil '
                "with its trapezoid vertical quadrature).")
        # NEMO's e3w(Kmm)-weighted trapezoid is NOT exclusive to hpg_sco: the
        # SAME -g/2 * e3w(Kmm) * (rhd(jk)+rhd(jk-1)) accumulation is the
        # z-coordinate recurrence too.
        #   hpg_zco  dynhpg.F90:270-271 (surface zcoef1 = zcoef0*e3w(..,1,Kmm))
        #            dynhpg.F90:284-291 (interior, e3w factored OUT of the
        #            horizontal difference because e3w is horizontally uniform
        #            in z-coordinates, and NO gdept_z0 stretching correction)
        #   hpg_sco  dynhpg.F90:343-350,366-374 (e3w carried INSIDE the
        #            horizontal difference + the zuap/zvap gdept_z0 term)
        # So the allow-list is widened -- not the guard removed -- to the two
        # schemes whose vertical recurrence NEMO actually pairs the trapezoid
        # with: "nemo_sco" (hpg_sco) and "adcroft" (this model's z-coordinate
        # / partial-cell PGF, the hpg_zco class).  Receipt for the adcroft
        # pairing: packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py's GYRE
        # card, where selecting the trapezoid moved the single-step HPG ratio
        # against NEMO's utrd_hpg from 0.9656 to 0.9702.  Adding a scheme here
        # requires the same: the oracle line showing it takes the e3w
        # trapezoid, and a receipt measuring it.
        _trapezoid_certified_pgf = {"nemo_sco", "adcroft"}
        if (getattr(config, "pgf_quadrature", "cell_integral")
                == "nemo_trapezoid"
                and pgf_scheme not in _trapezoid_certified_pgf):
            raise ValueError(
                'pgf_quadrature="nemo_trapezoid" is NEMO\'s dynhpg vertical '
                "recurrence and is certified only with pgf_scheme in "
                f"{sorted(_trapezoid_certified_pgf)} "
                "(hpg_sco dynhpg.F90:343-374, hpg_zco dynhpg.F90:270-296); "
                f"got pgf_scheme={pgf_scheme!r}")
        _eos_name = getattr(config, "eos", "linear")
        _eos_depth = getattr(config, "eos_depth", "insitu")
        if _eos_name == "nemo_teos10" and _eos_depth != "geometric":
            raise ValueError(
                'eos="nemo_teos10" requires eos_depth="geometric"; NEMO '
                "passes gdept directly to eosbn2")
        # NEMO's eos_insitu feeds the LIVE geometric gdept to the polynomial in
        # BOTH of its pressure-dependent arms, so both are certified for the
        # geometric depth ladder:
        #   np_teos10/np_eos80  eosbn2.F90:260  zh = gdept(ji,jj,jk,Knn)*r1_Z0
        #   np_seos             eosbn2.F90:297  zh = gdept(ji,jj,jk,Knn)
        # Receipts: docs/ocean/fidelity/testcases/nemo_testcases_l1_phase3_receipt.md:139
        # (nemo_teos10, LOCK/OVERFLOW -- the arm's provenance row; the number
        # behind it is at :73-79 of the same file, where pinning geometric took
        # the LOCK stage-1 u RHS error 5.6854e-9 -> 3.6863e-17) and
        # docs/ocean/fidelity/dino_tendency_certificate.md:15-16 (nemo_seos, DINO
        # nemo_paper/nemo_dino_kamm/_mlf; "required for the S-EOS thermobaric
        # depth term", max|drho'| 1.5e-5).  Adding an EOS here requires the same:
        # the oracle line showing it takes gdept, and a receipt measuring it.
        _geometric_certified_eos = {
            "nemo_eos80", "nemo_teos10", "nemo_seos",
        }
        if (_eos_depth == "geometric"
                and _eos_name not in _geometric_certified_eos):
            raise ValueError(
                'eos_depth="geometric" is certified only with eos in '
                f'{sorted(_geometric_certified_eos)} (NEMO eos_insitu passes '
                "live gdept in its np_teos10/np_eos80 arm, eosbn2.F90:260, and "
                "its np_seos arm, eosbn2.F90:297); got "
                f"eos={_eos_name!r}")
        _rdsm = getattr(config, "runoff_depth_spread_map", None)
        if _rdsm is not None:
            import numpy as _np
            _rdsm_np = _np.asarray(_rdsm)
            if (not _np.isfinite(_rdsm_np).all()) or (_rdsm_np <= 0).any():
                raise ValueError(
                    "runoff_depth_spread_map must be finite and > 0 "
                    "everywhere (build it with nemo_runoff_depth_map, "
                    "which floors at 1 m) — zero/negative cells would "
                    "silently drop the runoff dilution (codex r11 LOW).")
        _valid_pgf_quad = {"cell_integral", "nemo_trapezoid"}
        _pgf_quad = getattr(config, "pgf_quadrature", "cell_integral")
        if _pgf_quad not in _valid_pgf_quad:
            raise ValueError(
                f"pgf_quadrature must be one of {_valid_pgf_quad}, "
                f"got {_pgf_quad!r}")
        if getattr(config, "vorticity_scheme", "al81") == "een_planetary":
            # NEMO ln_dynvor_een under ln_dynadv_vec=.false. (dynvor.F90:874
            # routes np_EEN; dyn_vor_init:891-893 gives the flux-form arm
            # ntot = np_CME).  vor_een's np_CME branch (dynvor.F90:780-783) is
            # ff_f plus a metric term whose two coefficients are
            #   di_e2v_2e1e2f = (e2v(i+1,j) - e2v(i,j)) * 0.5 * r1_e1e2f
            #   dj_e1u_2e1e2f = (e1u(i,j+1) - e1u(i,j)) * 0.5 * r1_e1e2f
            # (dynvor.F90:905-908).  On a Cartesian mesh whose scale factors
            # are one repeated constant those differences are bitwise zero and
            # the branch IS np_COR.  On any other mesh they are not, and the
            # transcription would be silently incomplete -- so this scheme is
            # admitted only on a constant-scale-factor grid, and only with the
            # flux-form momentum it was read off.
            _cor_s = getattr(config, "coriolis_scheme", "matsuno_split")
            if _cor_s != "explicit_ab2":
                raise ValueError(
                    'vorticity_scheme="een_planetary" carries the planetary '
                    "Coriolis inside NEMO's vor_een triad, so the Matsuno "
                    'rotation must be off: requires coriolis_scheme='
                    f'"explicit_ab2", got {_cor_s!r}.')
            _ma = getattr(config, "momentum_advection", "vector_invariant")
            if _ma != "flux_form":
                raise ValueError(
                    'vorticity_scheme="een_planetary" transcribes NEMO\'s '
                    "FLUX-FORM vorticity arm only (dyn_vor_init:891-893). "
                    "Vector-invariant momentum routes NEMO to np_CRV, where "
                    "the triad also carries the RELATIVE vorticity — use "
                    '"een_total" for that. Got momentum_advection='
                    f"{_ma!r}.")
            if config.een_e3f_scheme != "nemo_avg4":
                raise ValueError(
                    'vorticity_scheme="een_planetary" needs NEMO\'s own '
                    'e3f_vor thickness: requires een_e3f_scheme="nemo_avg4", '
                    f"got {config.een_e3f_scheme!r}.")
            _emw = getattr(config, "een_metric_weighting", "off")
            if _emw != "nemo":
                raise ValueError(
                    'vorticity_scheme="een_planetary" needs vor_een\'s own '
                    "e1v/e2u transport weighting (dynvor.F90:791-792,804-806):"
                    f' requires een_metric_weighting="nemo", got {_emw!r}.')
            _bt_cor = getattr(config.barotropic, "barotropic_coriolis", "avg")
            _bt_spl = getattr(config, "barotropic_coriolis_split", "frozen")
            if _bt_cor != "een_metric" or _bt_spl != "live":
                raise ValueError(
                    'vorticity_scheme="een_planetary" requires the MATCHING '
                    "barotropic arm: NEMO's dyn_cor_2D_init runs the same "
                    "triad on ff_f/e3f_vor under np_EEN "
                    "(dynspg_ts.F90:1326-1345), and the depth-mean of the "
                    "baroclinic Coriolis must be subtracted with that same "
                    'stencil. Requires barotropic_coriolis="een_metric" and '
                    'barotropic_coriolis_split="live"; got '
                    f"{_bt_cor!r} / {_bt_spl!r}.")
            # The constant-scale-factor requirement is a property of the MESH,
            # which this config-only validator cannot see; it is enforced by
            # ``assert_een_planetary_metric_term_vanishes`` at the point the
            # card is built (nemo_testcase_recipe.py), where the grid is.
        if getattr(config, "vorticity_scheme", "al81") in (
                "ene_total", "een_total"):
            _vs = getattr(config, "vorticity_scheme", "al81")
            # ene_total = NEMO np_CRV (ENE Sadourny 2-point + vertex f);
            # een_total = NEMO ln_dynvor_een (AL81/EEN 12-point triad + vertex f,
            # the DINO form). Both carry the planetary Coriolis INSIDE the
            # vector-invariant vorticity flux, so the same three constraints hold.
            if getattr(config, "coriolis_scheme", "matsuno_split") != "explicit_ab2":
                raise ValueError(
                    f'vorticity_scheme="{_vs}" (planetary Coriolis inside the '
                    "vorticity flux) requires coriolis_scheme=\"explicit_ab2\" — "
                    "under matsuno_split the separate Matsuno rotation would "
                    "double-apply f. Got "
                    f'{getattr(config, "coriolis_scheme", "matsuno_split")!r}.')
            if getattr(config, "barotropic_coriolis_split", "frozen") == "live":
                # The live pre-step subtraction (NEMO dynspg_ts:296-300) must use
                # the SAME barotropic-Coriolis stencil as the in-substep dyn_cor_2D
                # so they cancel at the pre-step state. dyn_cor_2D_init switches
                # its coefficient build on the SAME nvor_scheme the 3-D vorticity
                # operator uses (dynspg_ts.F90:1326, fed by ln_dynvor_ene/een in
                # dynvor.F90:871-875): np_EEN builds 3-point triads
                # (dynspg_ts.F90:1327-1381), np_ENE builds 2-point Sadourny
                # coefficients (:1383-1410), and dyn_cor_2D (:1483-1506) applies
                # whichever set was built. So vorticity_scheme="ene_total"
                # requires barotropic_coriolis="ene"/"ene_metric" and
                # "een_total" requires "een"/"een_metric" -- the legacy 4-pt-avg
                # face-f "avg" stencil matches neither and would leave an O(1)
                # residual Coriolis.
                _required_bt = (("ene", "ene_metric") if _vs == "ene_total"
                                else ("een", "een_metric"))
                if getattr(config.barotropic, "barotropic_coriolis",
                           "avg") not in _required_bt:
                    raise ValueError(
                        f'vorticity_scheme="{_vs}" with '
                        'barotropic_coriolis_split="live" requires '
                        f"barotropic.barotropic_coriolis in {_required_bt!r}: "
                        "the _total planetary term and dyn_spg_ts::dyn_cor_2D "
                        "must use the SAME ENE/EEN transport-form flux, so the "
                        "live pre-step subtraction must use the SAME "
                        "stencil to cancel — the 4-pt-avg \"avg\" stencil "
                        "would not. Got barotropic_coriolis="
                        f'{getattr(config.barotropic, "barotropic_coriolis", "avg")!r}.')
            if getattr(config, "momentum_advection",
                       "vector_invariant") != "vector_invariant":
                raise ValueError(
                    f'vorticity_scheme="{_vs}" carries the planetary '
                    "Coriolis inside the VECTOR-INVARIANT vorticity flux; the "
                    "flux-form/WENO momentum branches never receive f_vtx, so "
                    "combining them would silently DROP f x u entirely "
                    "(review finding 2026-07-16). Got momentum_advection="
                    f'{getattr(config, "momentum_advection", "?")!r}.')
        _bt_split = getattr(config, "barotropic_coriolis_split", "frozen")
        if _bt_split not in ("frozen", "live"):
            raise ValueError(
                "Unknown barotropic_coriolis_split scheme: must be one of "
                f'("frozen", "live"), got {_bt_split!r}')
        if _bt_split == "live":
            if getattr(config, "coriolis_scheme", "matsuno_split") != "explicit_ab2":
                raise ValueError(
                    'barotropic_coriolis_split="live" requires '
                    'coriolis_scheme="explicit_ab2" (the frozen F_slow must '
                    "contain the planetary Coriolis depth-mean for the "
                    "pre-step subtraction to be exact). Got coriolis_scheme="
                    f'{getattr(config, "coriolis_scheme", "matsuno_split")!r}.')
            if config.barotropic.barotropic_solver != "explicit_substep":
                raise ValueError(
                    'barotropic_coriolis_split="live" is implemented for the '
                    'explicit_substep barotropic solver only; got '
                    f"{config.barotropic.barotropic_solver!r}.")
            if getattr(config.barotropic, "barotropic_slow_forcing_ab2", False):
                raise ValueError(
                    'barotropic_coriolis_split="live" is incompatible with '
                    "barotropic_slow_forcing_ab2=True: the AB2-blended F_slow "
                    "Coriolis (1.5*cor^n - 0.5*cor^(n-1)) would not cancel the "
                    "single-time pre-step subtraction, leaving a transient "
                    "residual planetary Coriolis in the forcing.")
            # nn_bt_flt=3 deliberately applies live Coriolis to the AB3
            # mid-step velocity after subtracting the plain Kmm value from the
            # frozen forcing (GYRE's key_RK3 branch: dynspg_ts.F90:296,689 —
            # NOT the MLF branch's :359, which this build never compiles;
            # see provenance/cpp_GYRE_OMIP_L2_P3.fcm: key_RK3).  Their
            # non-cancellation is the intended AB3 evolution, not an
            # incompatibility; at a cold start (LN_RSTART=F) ll_init=.TRUE.
            # at kt==nit000, so jn=1's extrapolation coefficients are za1=1,
            # za2=za3=0 (dynspg_ts.F90:535-538) and un_e is seeded from
            # puu_b(:,:,Kmm) under LN_BT_FW=T (:487) — substep 1 DOES cancel
            # bit-exactly; substep>=2 and every later kt do not, which is the
            # AB3 evolution NEMO always runs (the za1/za2/za3 block is not
            # gated on nn_bt_flt).
        _valid_time_int = {"euler", "ab2", "rk3", "rk3_ws"}
        if config.tracer_time_integrator not in _valid_time_int:
            raise ValueError(
                f"tracer_time_integrator must be one of {_valid_time_int}, "
                f"got {config.tracer_time_integrator!r}")
        _mom_ti_for_ws = getattr(config, "momentum_time_integrator", "euler")
        # NEMO's carried after-SSH slot belongs to the RK3 stage program: it is
        # written by that program's end-of-step time-level rotation
        # (stprk3.F90:217, raw source) and the leapfrog fills the same slot from
        # continuity instead.  This is NOT an inference about which form a card
        # should take -- the card STATES that -- it is a CONSISTENCY CHECK
        # between two fields the card states, in the same shape as the rk3_ws
        # pairing check immediately below, and it raises rather than quietly
        # running a program that would read the slot without ever writing it.
        if (getattr(config, "nemo_first_wzv_after_ssh", "")
                == "rk3_extrapolated_carried"):
            if _mom_ti_for_ws != "rk3_ws":
                raise ValueError(
                    "nemo_first_wzv_after_ssh='rk3_extrapolated_carried' is "
                    "NEMO's RK3 end-of-step extrapolation and only the RK3 "
                    "stage program writes that slot; this card states "
                    f"momentum_time_integrator={_mom_ti_for_ws!r}. Select "
                    "rk3_ws, or state the after-SSH form that card's own "
                    "time-stepping program leaves behind")
            _post = [name for name, on in (
                ("polar_filter.use_polar_filter",
                 config.polar_filter.use_polar_filter),
                ("freeze_floor", config.freeze_floor),
                ("ew_cyclic_overlap",
                 getattr(config, "ew_cyclic_overlap", False)),
            ) if on]
            if _post:
                raise ValueError(
                    "nemo_first_wzv_after_ssh='rk3_extrapolated_carried' "
                    "carries NEMO's own after-SSH slot, built from the sea "
                    "surface the step produced; "
                    f"{_post} rewrite that sea surface AFTER the step, so the "
                    "slot the next step reads would not match the height it "
                    "enters with. NEMO applies none of these")
        if ((config.tracer_time_integrator == "rk3_ws")
                != (_mom_ti_for_ws == "rk3_ws")):
            raise ValueError(
                "NEMO rk3_ws is one coupled momentum/tracer stage program; "
                "select rk3_ws for both integrators or for neither")
        if config.tracer_time_integrator == "rk3_ws":
            if config.tracer_advection != "fct2":
                raise ValueError(
                    "the certified NEMO rk3_ws scheme identity requires "
                    "tracer_advection='fct2'; FCT4/PPM is not certified")
            # UNION: the L2 GYRE card runs the vector-invariant/ENE arm of
            # the same WS-RK3 identity, so rk3_ws admits TWO complete momentum
            # programs.  The flux-form arm carries the iso branch's UP3
            # selector split (bare "upwind3" is refused; the NEMO sign
            # convention is "nemo_up3", Oceananigans' is "oceananigans_up3").
            _ws_flux_up3 = (
                config.momentum_advection == "flux_form"
                and config.momentum_flux_scheme == "nemo_up3"
                and getattr(config, "vertical_momentum_scheme",
                            "upwind_perturbation") == "nemo_up3"
            )
            _ws_vector_ene_c2 = (
                config.momentum_advection == "vector_invariant"
                and getattr(config, "vorticity_scheme", "al81") == "ene_total"
                and getattr(config, "ke_gradient_scheme", "centered") == "c2"
                and getattr(config, "vertical_momentum_scheme",
                            "upwind_perturbation") == "nemo_advective"
                and not getattr(config, "adaptive_implicit_vertadv", False)
            )
            # Same shared WS-RK3 vector program, with dynvor's EEN selector.
            # ORCA2 resolves ln_dynvor_een=T, nn_dynkeg=0 and ln_zad_Aimp=F
            # (dynvor.F90:1326-1332; stprk3_stg.F90:283-299).  All EEN
            # arithmetic already lives in the canonical vector-invariant
            # operator; this arm only admits the source-valid composition.
            _ws_vector_een_c2 = (
                config.momentum_advection == "vector_invariant"
                and getattr(config, "vorticity_scheme", "al81") == "een_total"
                and getattr(config, "ke_gradient_scheme", "centered") == "c2"
                and getattr(config, "vertical_momentum_scheme",
                            "upwind_perturbation") == "nemo_advective"
                and not getattr(config, "adaptive_implicit_vertadv", False)
            )
            if not (_ws_flux_up3 or _ws_vector_ene_c2 or _ws_vector_een_c2):
                raise ValueError(
                    "NEMO rk3_ws requires one complete momentum program: "
                    "flux_form/nemo_up3/nemo_up3 or "
                    "vector_invariant/(ene_total|een_total)/c2/"
                    "nemo_advective")
            if getattr(config, "outer_integrator", "forward_euler") != "forward_euler":
                raise ValueError(
                    "NEMO rk3_ws does not compose with a second outer "
                    "integrator")
            _ws_gm_redi = getattr(config, "gm_redi", None)
            if (_ws_gm_redi is not None
                    and getattr(_ws_gm_redi, "kappa_GM", 0.0) != 0.0):
                raise ValueError(
                    "NEMO rk3_ws does not yet support staged GM bolus "
                    "transports; pure Redi (kappa_GM=0) is supported")
        if (config.momentum_flux_scheme == "nemo_up3"
                and _mom_ti_for_ws != "rk3_ws"):
            # S-47 (docs/ocean/fidelity/nemo_branch_isomorphism_map.md):
            # momentum_flux_scheme="nemo_up3" selects NEMO's dyn_adv_up3
            # T-point upwind SELECTOR (dynadv_up3.F90:166,169-170, the
            # advected-velocity pair) on its own -- 9bbf9f7bd keys that
            # selector by the reference the caller names, not the time
            # integrator. But NEMO's own face THICKNESS for that same
            # routine, e3u(Kmm) (dynadv_up3.F90:160,205-207), is wired only
            # inside the momentum_time_integrator="rk3_ws" stage program
            # (60d0c5420, momentum_flux_face_thickness from
            # _nemo_ws_qco_stage_faces); every other integrator falls back
            # to the legacy min-of-stretched-T-thickness rule S-47 measured
            # as a first-order-wrong reconstruction of that same macro. So
            # nemo_up3 on any integrator other than rk3_ws gets NEMO's
            # selector without NEMO's face thickness -- a pairing NEMO
            # itself never runs.
            raise ValueError(
                "momentum_flux_scheme='nemo_up3' (NEMO dyn_adv_up3's "
                "T-point upwind selector) requires "
                "momentum_time_integrator='rk3_ws' -- the only lane that "
                "wires NEMO's own e3u(Kmm) face thickness (S-47) into that "
                f"selector; got momentum_time_integrator={_mom_ti_for_ws!r}"
                ", which would pair NEMO's selector with the legacy "
                "min-of-stretched-face-thickness rule NEMO does not use. "
                "Select momentum_time_integrator='rk3_ws' for NEMO's full "
                "nemo_up3 program, or use "
                "momentum_flux_scheme='oceananigans_up3' (the "
                "transport-selector arm) on this integrator.")
        if config.bbl_adv_option not in (0, 2):
            raise ValueError("bbl_adv_option must be 0 or 2")
        if config.bbl_diffusive_option not in (0, 1):
            raise ValueError("bbl_diffusive_option must be 0 or 1")
        if config.bbl_adv_option == 2 and config.bbl_gamma_s <= 0.0:
            raise ValueError("bbl_adv_option=2 requires bbl_gamma_s > 0")
        if (config.bbl_diffusive_option == 1
                and config.bbl_aht_m2_s <= 0.0):
            raise ValueError(
                "bbl_diffusive_option=1 requires bbl_aht_m2_s > 0")
        if (config.bbl_diffusive_option == 1
                and config.tracer_time_integrator != "rk3_ws"):
            raise ValueError(
                f"bbl_diffusive_option=1 is not honoured by "
                f"tracer_time_integrator={config.tracer_time_integrator!r}: "
                "NEMO tra_bbl_dif is wired at RK stage 3 only on the shared "
                "tracer_time_integrator='rk3_ws' identity")
        if (config.bbl_adv_option == 2
                and config.tracer_time_integrator != "rk3_ws"):
            # S-42 (docs/ocean/fidelity/nemo_branch_isomorphism_map.md): the
            # in-stage BBL hook is built ONLY inside the rk3_ws tracer lane
            # (elif _tti == "rk3_ws": ... bbl_transports/
            # apply_bbl_adv_tendency folded into the stage-3 tracer RHS).
            # euler/ab2/rk3 never build that hook, so bbl_adv_option=2 there
            # resolves and is silently never read -- no boundary layer, no
            # warning. Fail fast instead.
            raise ValueError(
                f"bbl_adv_option=2 is not honoured by "
                f"tracer_time_integrator={config.tracer_time_integrator!r}: "
                "the in-model BBL hook is built only on "
                "tracer_time_integrator='rk3_ws'; on every other tracer lane "
                "this option resolves but is never read, silently dropping "
                "the boundary layer. Either select tracer_time_integrator="
                "'rk3_ws', or leave bbl_adv_option=0 and apply BBL outside "
                "the model via legoesm.ocean.physics.bbl_adv."
                "apply_bbl_adv_step (the driver-side path run_omip_core2.py "
                "uses via --bbl-adv).")
        if getattr(config, "store_salt_flux", False):
            # Refuse-not-ignore: the capture stores "the flux the model
            # applied", which is only well-defined per step on the euler
            # path with a scheme that exposes its horizontal fluxes.  AB2
            # extrapolates two levels' divergences, RK3 combines stages,
            # FCT/multidim form fluxes inside their kernels, SOM advects
            # moments -- storing any single-stage pair there would be a lie
            # with the right shape.
            if config.tracer_time_integrator != "euler":
                raise ValueError(
                    "store_salt_flux requires tracer_time_integrator="
                    f"'euler' (got {config.tracer_time_integrator!r}): under "
                    "AB2/RK3 the applied salt flux is a multi-level/staged "
                    "combination the capture would misrepresent.")
            _outer_for_salt = getattr(config, "outer_integrator",
                                      "forward_euler")
            if _outer_for_salt != "forward_euler":
                # codex round-1 RED 2: the OUTER AB2 blends the FULL explicit
                # tracer increment (S_incr_prev) across steps -- the applied
                # update is a_n*dS^n - a_p*dS^{n-1} even with the inner
                # integrator at euler, so the per-step captured pair is NOT
                # the applied flux.  Leapfrog applies 2*dt*F^n to the BEFORE
                # state (a different base and weight).  Refuse both.
                raise ValueError(
                    "store_salt_flux requires outer_integrator="
                    f"'forward_euler' (got {_outer_for_salt!r}): the outer "
                    "AB2/leapfrog combine explicit tracer increments across "
                    "time levels, so a single-step captured flux would not "
                    "be the applied one.")
            if config.tracer_advection in ("ppm_fct", "fct2",
                                           "dst3_multidim", "som"):
                raise ValueError(
                    "store_salt_flux is not supported with tracer_advection="
                    f"{config.tracer_advection!r}: the scheme does not expose "
                    "its horizontal face fluxes (FCT/multidim form them "
                    "in-kernel; SOM advects moments).  Use upwind/tvd/"
                    "superbee/centered/ppm/dst3/weno5/weno7.")
        if config.ab2_epsilon < 0.0:
            raise ValueError(
                f"ab2_epsilon must be >= 0, got {config.ab2_epsilon!r}")
        _valid_mom_int = {"euler", "rk3", "rk3_ws"}
        _mom_ti = getattr(config, "momentum_time_integrator", "euler")
        if _mom_ti not in _valid_mom_int:
            raise ValueError(
                f"momentum_time_integrator must be one of {_valid_mom_int}, "
                f"got {_mom_ti!r}")
        # Implicit surface-forcing placement (Veros) requires the implicit
        # vertical-mixing solve — that is where the surface TRACER source is
        # added (the backward-Euler RHS).  With explicit vertical mixing there
        # is no implicit solve to host it; reject the combination rather than
        # silently applying the forcing explicitly (dispatch discipline).
        if config.surface_forcing_implicit and not config.implicit_vertical_mixing:
            raise ValueError(
                "surface_forcing_implicit=True requires "
                "implicit_vertical_mixing=True: the surface TRACER forcing "
                "(restoring + q_net + shortwave) is applied at weight 1.0 inside "
                "the backward-Euler vertical-mixing solve (Veros's "
                "core/thermodynamics.py placement). With explicit vertical "
                "mixing there is no implicit solve to host the source. Set "
                "implicit_vertical_mixing=True, or surface_forcing_implicit=False "
                "to keep the explicit surface-forcing placement.")

        # Implicit SPONGE placement (EXT-N2, Veros tempsalt_sources) likewise
        # lives in the backward-Euler vertical-mixing solve; with explicit
        # vertical mixing there is no implicit solve to host the source —
        # reject rather than silently dropping the sponge or silently applying
        # it explicitly (dispatch discipline).
        if (getattr(config, "sponge_forcing_implicit", False)
                and not config.implicit_vertical_mixing):
            raise ValueError(
                "sponge_forcing_implicit=True requires "
                "implicit_vertical_mixing=True: the sponge T/S relaxation is "
                "applied at weight 1.0 inside the backward-Euler "
                "vertical-mixing solve (Veros's tempsalt_sources placement, "
                "core/thermodynamics.py:419 -> core/diffusion.py:132-141). "
                "Set implicit_vertical_mixing=True, or "
                "sponge_forcing_implicit=False to keep the explicit "
                "stage-10c sponge placement.")

        # Veros u_centered dzw slot for the implicit vertical-diffusion solves
        # lives INSIDE the backward-Euler tracer/friction solve (it picks the
        # gradient divisor there); with explicit vertical mixing there is no
        # implicit solve to host it — reject rather than silently ignoring the
        # flag (dispatch discipline).
        if (getattr(config, "implicit_vmix_dzw_slot", False)
                and not config.implicit_vertical_mixing):
            raise ValueError(
                "implicit_vmix_dzw_slot=True requires "
                "implicit_vertical_mixing=True: the Veros dzw gradient slot is "
                "the divisor of the backward-Euler tracer/momentum-friction "
                "vertical-diffusion solve (thermodynamics.py:267 "
                "delta = dt·kappaH/dzw). With explicit vertical mixing there is "
                "no implicit solve to host it. Set implicit_vertical_mixing=True, "
                "or implicit_vmix_dzw_slot=False to keep the midpoint slot.")

        # The NEMO e3w(Kmm) divisor is no longer a flag: it belongs to the
        # NEMO identity (zdf_implicit_solver_evaluation="nemo_literal"), which
        # selects the literal dyn_zdf/tra_zdf program it lives in.  The old
        # mutual-exclusion raise against `implicit_vmix_e3t_now_divisor` goes
        # with the field.  It is NOT retargeted onto the identity: that would
        # make a previously-tolerated combination a hard error, no card in the
        # tree selects both (`veros_faithful_v1`, the only dzw consumer, runs
        # `shared_thomas`), and it would break the committed divisor probe that
        # plants a coordinate through the dzw slot.  Precedence is explicit and
        # documented at the divisor block instead: the Veros dzw slot is
        # checked FIRST, so a config selecting both gets Veros's divisor.

        # Additive momentum vertical-friction placement (Veros solve_stream.py)
        # is defined relative to the AB2 outer integrator (the increment is
        # added alongside the AB2-extrapolated explicit tendency) and needs the
        # implicit vertical-friction solve to produce that increment.  Reject
        # unsupported combinations rather than silently falling back to the
        # sequential placement (dispatch discipline).
        if config.momentum_friction_additive:
            if not config.implicit_vertical_mixing:
                raise ValueError(
                    "momentum_friction_additive=True requires "
                    "implicit_vertical_mixing=True: the additive increment IS "
                    "the backward-Euler vertical-friction solve evaluated on "
                    "the pre-step velocity u^n (Veros core/friction.py). With "
                    "explicit vertical mixing there is no implicit friction "
                    "solve to relocate.")
            if config.outer_integrator != "ab2":
                raise ValueError(
                    "momentum_friction_additive=True requires "
                    'outer_integrator="ab2": the Veros placement adds the '
                    "u^n-evaluated friction increment alongside the "
                    "AB2-extrapolated explicit tendency (solve_stream.py). "
                    f"Got outer_integrator={config.outer_integrator!r}; only "
                    "the AB2 path was oracle-verified for this placement.")

        # Coriolis time-stepping placement (Veros explicit-tendency AB2 vs the
        # default Matsuno rotation sub-step).  "explicit_ab2" routes the plain
        # f×u through du_dt (so the outer AB2 extrapolates it and its depth-mean
        # feeds the barotropic rigid-lid slow forcing) and skips both the Matsuno
        # sub-step and the barotropic solver's own Coriolis addition.  Reject
        # unsupported combinations rather than silently mis-placing Coriolis.
        _cor_scheme = getattr(config, "coriolis_scheme", "matsuno_split")
        if _cor_scheme not in VALID_CORIOLIS_SCHEME:
            raise ValueError(
                f"coriolis_scheme must be one of "
                f"{sorted(VALID_CORIOLIS_SCHEME)}, got {_cor_scheme!r}")
        if _cor_scheme == "explicit_ab2":
            # The explicit f×u tendency needs a stably-rotating outer integrator.
            # AB2(-eps) has a stable rotation region; SSP-RK3 does too (|G|<=1 up to
            # f·dt~sqrt(3)), and — unlike operator-splitting the Coriolis AFTER the
            # RK3 PGF stages — it keeps PGF and Coriolis COUPLED inside the stages,
            # holding geostrophic balance (the fix for the O(dt^2) split-growth of
            # the GYRE forced current). Forward-Euler alone is unstable (|G|>1).
            if config.outer_integrator not in (
                    "ab2", "leapfrog", "nemo_mlf") and getattr(
                    config, "momentum_time_integrator",
                    "euler") not in ("rk3", "rk3_ws"):
                raise ValueError(
                    'coriolis_scheme="explicit_ab2" requires '
                    'outer_integrator in ("ab2","leapfrog","nemo_mlf") OR '
                    'momentum_time_integrator in ("rk3", "rk3_ws"): '
                    "an explicit forward-Euler Coriolis at weight 1.0 is "
                    "unconditionally UNSTABLE for pure rotation "
                    "(|G|=sqrt(1+(f·dt)²)>1); AB2(-eps), leapfrog/nemo_mlf "
                    "(neutral, |G|=1 for f·dt<1, computational mode damped by "
                    "the Robert-Asselin filter) or SSP-RK3 have a stable "
                    "rotation region. Got outer_integrator="
                    f"{config.outer_integrator!r}, momentum_time_integrator="
                    f"{getattr(config, 'momentum_time_integrator', 'euler')!r}.")
            if config.barotropic.barotropic_solver not in (
                    "rigid_lid", "implicit_cn", "implicit_unsplit",
                    "explicit_substep"):
                raise ValueError(
                    'coriolis_scheme="explicit_ab2" requires '
                    'barotropic_solver in ("rigid_lid","implicit_cn",'
                    '"implicit_unsplit","explicit_substep"): '
                    'the explicit Coriolis '
                    "tendency reaches the barotropic mode through its depth-mean "
                    "in the slow forcing F_slow (= Veros solve_stream.py uloc/"
                    "vloc, the depth-integral of du including Coriolis), and the "
                    "solver's own f×U_bt addition is gated off to avoid "
                    "double-counting (rigid_lid / explicit_substep: "
                    "add_barotropic_coriolis=False; "
                    "implicit_cn: _cori_fac=0 in the FB predictor). The "
                    "explicit_substep solver now gates its in-substep Coriolis "
                    "off too (Oceananigans split-explicit convention: ∂_tU = "
                    "−gH∇η + G^U, no in-substep Coriolis), which removes the "
                    "C-grid 4-point Coriolis rotational null mode. Got "
                    f"barotropic_solver={config.barotropic.barotropic_solver!r}.")
            if getattr(config, "coriolis_energy_conserving", False):
                raise ValueError(
                    'coriolis_scheme="explicit_ab2" is incompatible with '
                    "coriolis_energy_conserving=True: under explicit_ab2 the "
                    "planetary Coriolis enters du_dt via the FACE-f coriolis_cgrid "
                    "and its depth-mean is carried into the barotropic predictor "
                    "through F_slow. The implicit_cn solver then gates its own FB "
                    "Coriolis off (_cori_fac=0) only in the face-f branch; the "
                    "VERTEX-f energy-conserving branch is NOT gated, so enabling it "
                    "here would both double-count the barotropic Coriolis and mix a "
                    "vertex-f barotropic term with a face-f du_dt term (physically "
                    "inconsistent). Use coriolis_energy_conserving=False with "
                    "explicit_ab2 (the MITgcm-faithful face-f form), or switch to "
                    "the Matsuno split scheme for the vertex-f energy-conserving "
                    "Coriolis.")

        # barotropic_slow_forcing_ab2 (AB2 time-centering of the depth-mean
        # barotropic forcing F_slow) is only well-posed alongside the
        # explicit_ab2 Coriolis routing, and only with ab2_scope="total".
        # Validate on the static config at construction (dispatch-hardening):
        # reject the silent-misconfiguration combinations rather than producing
        # a Coriolis-free or time-inconsistent barotropic forcing.
        if getattr(config.barotropic, "barotropic_slow_forcing_ab2", False):
            if _cor_scheme != "explicit_ab2":
                raise ValueError(
                    'barotropic_slow_forcing_ab2=True requires '
                    'coriolis_scheme="explicit_ab2": the AB2 time-centering of '
                    "the barotropic slow forcing only carries the planetary "
                    "Coriolis to the barotropic mode when f×u enters du_dt "
                    "(explicit_ab2, whose depth-mean lands in F_slow). Under "
                    "matsuno_split the in-substep f·V_at_u 4-point average (the "
                    "2Δx rotational null mode) is still active AND F_slow has no "
                    "Coriolis, so the flag would AB2-extrapolate a Coriolis-free "
                    "forcing while the null mode persists — a silent no-op cure. "
                    f"Got coriolis_scheme={_cor_scheme!r}.")
            if getattr(config, "ab2_scope", "total") == "advective":
                raise ValueError(
                    'barotropic_slow_forcing_ab2=True is not supported with '
                    'ab2_scope="advective": the stored AB2 prev is F_slow BEFORE '
                    "the du_diss depth-mean is folded in, so under advective "
                    "scope the dissipative depth-mean would be applied "
                    "un-time-centered while the prev omits it (an inconsistent "
                    'AB2). Use ab2_scope="total" (dissipative terms then enter '
                    "du_dt and AB2-extrapolate with everything; §5 uses total).")

        # AB2 extrapolation scope (Veros-faithful dissipative placement). The
        # "advective" scope withholds the dissipative tendencies from the AB2
        # extrapolation and applies them at weight 1.0; it is meaningless
        # without the AB2 outer integrator (the FE path doesn't extrapolate, so
        # "total" and "advective" already agree). Reject unsupported
        # combinations rather than silently ignoring the scope.
        _ab2_scope = getattr(config, "ab2_scope", "total")
        if _ab2_scope not in VALID_AB2_SCOPE:
            raise ValueError(
                f"ab2_scope must be one of {sorted(VALID_AB2_SCOPE)}, "
                f"got {_ab2_scope!r}")
        if _ab2_scope == "advective" and config.outer_integrator != "ab2":
            raise ValueError(
                'ab2_scope="advective" requires outer_integrator="ab2": the '
                "advective scope splits the explicit increment into an "
                "AB2-extrapolated advective part and a weight-1.0 dissipative "
                "part (momentum lateral friction + bottom drag; tracer lateral "
                "diffusion + GM/Redi), which is only meaningful for the AB2 "
                "outer integrator. The forward-Euler path applies every "
                "tendency at weight 1.0 already (no extrapolation to scope). "
                f"Got outer_integrator={config.outer_integrator!r}.")

        # Prescribed-flow science lever (vertical-physics isolation).  Fail-fast
        # value validation (dispatch discipline: static literal at construction)
        # + rejection of the one outer path whose tracer advection does NOT go
        # through the shared _step_impl mass-flux block.
        _pflow = getattr(config, "prescribed_flow", None)
        _valid_pflow = (None, "zero", "frozen")
        if _pflow not in _valid_pflow:
            raise ValueError(
                f"prescribed_flow must be one of {_valid_pflow}, got "
                f"{_pflow!r}")
        if (_pflow is not None
                and config.barotropic.barotropic_solver == "implicit_unsplit"):
            raise ValueError(
                'prescribed_flow is not supported with barotropic_solver='
                '"implicit_unsplit": that path advects tracers via the '
                "advective-form tendency inside _unsplit_ab2_step (no shared "
                "mass-flux block to pin), so the lever would silently NOT "
                "isolate the circulation. Use the split solvers "
                '("explicit_substep", "implicit_cn", "rigid_lid").')
        if _pflow is not None:
            # Parameterized ADVECTIVE tracer transports bypass the pinned
            # mass-flux block (codex findings 2+3): GM bolus/skew transport,
            # the prognostic-EKE closure (which also advects EKE with
            # state.u/v), and the Fox-Kemper MLE bolus tendency all move
            # tracers with an eddy-induced velocity that the lever does NOT
            # pin — they would silently re-inject circulation into the
            # "isolated" run.  Reject every config expression of them at
            # construction rather than silently running a leaky isolation.
            if config.gm_redi is not None:
                raise ValueError(
                    "prescribed_flow cannot be combined with GM/Redi "
                    "(config.gm_redi is set): the GM bolus (skew) transport "
                    "is parameterized tracer ADVECTION applied outside the "
                    "pinned mass-flux block, and the prognostic-EKE closure "
                    "(gm_redi.eke) additionally advects EKE with the model "
                    "u/v — the lever would not isolate the circulation. "
                    "Disable GM/Redi (gm_redi=None) for the isolation run.")
            _phys = config.physics
            if _phys is not None:
                _lat_mix = getattr(_phys, "lateral_mixing", None)
                if (_lat_mix is not None
                        and getattr(_lat_mix, "scheme", "none") == "gm_redi"):
                    raise ValueError(
                        "prescribed_flow cannot be combined with the physics-"
                        'pipeline GM/Redi (physics.lateral_mixing.scheme='
                        '"gm_redi"): bolus transport is parameterized tracer '
                        "ADVECTION injected through physics_fn, bypassing the "
                        "pinned mass-flux block. Disable GM/Redi "
                        '(lateral_mixing scheme "none") for the isolation run.')
                if getattr(_phys, "mle", None) is not None:
                    raise ValueError(
                        "prescribed_flow cannot be combined with Fox-Kemper "
                        "MLE (config.physics.mle is set): the MLE "
                        "restratification is a bolus tracer transport "
                        "injected through physics_fn before the circulation "
                        "pin — parameterized ADVECTION the lever does not "
                        "isolate. Disable MLE (mle=None) for the isolation "
                        "run.")

        return config

    def check_barotropic_cfl(self, dt: float) -> float:
        """Check barotropic CFL and warn if marginal or unstable.

        Parameters
        ----------
        dt : float
            Baroclinic timestep [s].

        Returns
        -------
        cfl : float
            Barotropic CFL number.
        """
        import math
        import warnings

        g = self.config.g
        H_max = self.z_coord.H_max
        n_sub = self.config.barotropic.n_barotropic_substeps
        # grid.dx and grid.dy are "distance over 2 cells", so cell width = dx/2.
        # grid.dy is (n_lat,) — take the global min (Mercator-safe).
        dx_min = min(
            float(jnp.min(self.grid.dx)) / 2.0,
            float(jnp.min(self.grid.dy)) / 2.0,
        )

        c_baro = math.sqrt(g * H_max)
        dt_baro = dt / n_sub
        cfl = c_baro * dt_baro / dx_min

        if cfl > 0.8:
            n_min = math.ceil(c_baro * dt / (0.8 * dx_min))
            warnings.warn(
                f"Barotropic CFL = {cfl:.2f} (> 0.8) — may be unstable. "
                f"c_baro={c_baro:.1f} m/s, dx_min={dx_min:.0f} m, "
                f"dt_baro={dt_baro:.1f} s. "
                f"Suggest n_barotropic_substeps >= {n_min} "
                f"(currently {n_sub}).",
                stacklevel=2,
            )

        # AB2 has a tighter advective CFL limit (~0.72 with eps=0.1)
        # than forward Euler (~1.0).  Warn if the advective CFL is
        # close to the AB2 stability boundary.
        if self.config.tracer_time_integrator == "ab2":
            # Rough advective CFL estimate using max barotropic velocity
            adv_cfl = c_baro * dt / dx_min  # upper bound (uses gravity wave speed)
            if adv_cfl > 0.7:
                warnings.warn(
                    f"Advective CFL estimate ~{adv_cfl:.2f} is near the "
                    f"AB2 stability limit (~0.72 with eps={self.config.ab2_epsilon}). "
                    f"Consider reducing dt or using tracer_time_integrator='euler'.",
                    stacklevel=2,
                )

        return cfl

    def check_coriolis_stability(self, dt: float) -> float:
        """Check the explicit-AB2 Coriolis stability margin and warn if marginal.

        Only meaningful for ``coriolis_scheme="explicit_ab2"`` (a no-op returning
        the f·dt_mom number otherwise).  An EXPLICIT Adams-Bashforth-2(-ε)
        Coriolis is conditionally stable in ``f·dt_mom``: the inertial-mode
        amplification |G| exceeds 1 (anti-damping) once ``f·dt_mom`` crosses ≈0.5,
        and the forced–damped solution diverges for the channel's available
        friction once ``f·dt_mom`` reaches ≈0.55–0.6 (verified numerically; see
        ``.physics-validator/coriolis_scheme/``).  Veros runs the ACC stably
        because its domain (|lat| ≤ 44°) keeps ``|f|max·dt_mom ≈ 0.49`` (|G|≈0.997,
        marginally below 1) and its vertical/bottom friction + AB_eps=0.1 hold the
        weak anti-damping bounded.  A global lat-lon ocean reaching |lat| ≥ 50°
        would EXCEED the margin; warn so the caller raises ``dt_mom_ratio`` (lowers
        dt_mom), restricts the domain, or stays on the default ``matsuno_split``.

        Parameters
        ----------
        dt : float
            Baroclinic clock timestep [s] (dt_tracer); the Coriolis is integrated
            with ``dt_mom = dt / dt_mom_ratio``.

        Returns
        -------
        fdt_max : float
            ``|f|max · dt_mom`` — the explicit-Coriolis stability number.
        """
        import warnings

        dt_mom = dt / self.config.dt_mom_ratio
        f_max = float(jnp.max(jnp.abs(self.grid.f)))
        fdt_max = f_max * dt_mom
        if getattr(self.config, "coriolis_scheme", "matsuno_split") != "explicit_ab2":
            return fdt_max
        # 0.5 = onset of inertial anti-damping (|G|>1); 0.55 = forced–damped
        # divergence threshold for the ACC's friction.  Warn at 0.5 (marginal),
        # strongly at 0.55 (likely unstable).
        if fdt_max > 0.55:
            warnings.warn(
                f"coriolis_scheme='explicit_ab2': |f|max·dt_mom = {fdt_max:.3f} "
                f"(> 0.55) — the explicit AB2-ε Coriolis is likely UNSTABLE "
                f"(inertial |G| > 1 and the forced–damped mode diverges for "
                f"typical ocean friction). f_max={f_max:.3e} s^-1, "
                f"dt_mom={dt_mom:.0f} s. Reduce dt_mom (raise dt_mom_ratio), "
                f"restrict the domain to |lat| < ~50°, or use "
                f"coriolis_scheme='matsuno_split'.",
                stacklevel=2,
            )
        elif fdt_max > 0.5:
            warnings.warn(
                f"coriolis_scheme='explicit_ab2': |f|max·dt_mom = {fdt_max:.3f} "
                f"(> 0.5) — marginal: the inertial mode is weakly anti-damped "
                f"(|G| slightly > 1), bounded only by friction + AB_eps. Acceptable "
                f"for the ACC (|f|max·dt_mom ≈ 0.49) but verify boundedness if the "
                f"domain extends poleward of ~44°.",
                stacklevel=2,
            )
        return fdt_max

    def tendencies(self, state: LatLonCGridOceanState, surface_forcing=None,
                   sponge=None, dt=300.0, momentum_only=False,
                   precomputed_geom_density=None, *, grid=None,
                   vertex_mask=None, skip_lateral_viscosity=False,
                   ab2_scope_override: str | None = None,
                   ldf_state=None, z_coord=None, config=None,
                   zad_continuity_dt=None,
                   zad_freshwater_eta_tendency=None,
                   momentum_flux_transport_velocity=None,
                   up3_upwind_selector=None,
                   momentum_flux_face_thickness=None,
                   ldf_thickness_operands=None,
                   ldf_metric_reciprocal_operands=None,
                   ene_metric_reciprocals=None,
                   ene_generic_f_vtx=False,
                   legacy_hpg_algebraic=False,
                   nemo_operator_association=False,
                   return_nemo_operator_components=False,
                   nemo_stage_zad_operands=None,
                   nemo_stage_zad_operand_observer=None,
                   nemo_stage_zad_eta_after_override=None):
        """Compute baroclinic tendencies.

        ``momentum_only=True`` skips the (T/S-frozen) tracer-diffusion
        tendency for the RK3 momentum sub-stages, which discard dT_dt/dS_dt
        — bit-identical du_dt/dv_dt, fewer halos/compute (see
        ``latlon_cgrid_ocean_baroclinic_tendencies``).

        ``precomputed_geom_density`` (a ``(J, h_k, rho_prime,
        p_prime_filled, nemo_hpg_rhd)`` tuple from
        :func:`compute_frozen_geom_density`)
        skips the EOS + baroclinic-pressure recompute (stages 1-3, frozen
        across RK3 momentum sub-stages) — bit-identical, ~the dominant
        per-substage cost (#25).

        ``grid``/``vertex_mask`` (optional, SPMD): when ``None`` (the
        production single-device path) the model's own ``self.grid`` /
        ``self._vertex_mask`` are used and behaviour is bit-identical to
        before; a future ``shard_map`` wrapper passes a per-device
        band-local grid (+ vertex mask) instead.

        ``ldf_state`` (private, ``nemo_mlf`` P1): forwarded verbatim to
        :func:`latlon_cgrid_ocean_baroclinic_tendencies` — see its docstring.
        ``None`` (every existing caller) ⇒ bit-identical.
        """
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        _grid = grid if grid is not None else self.grid
        _vmask = vertex_mask if vertex_mask is not None else self._vertex_mask
        # ``ab2_scope_override`` (leap-frog Nbb-diffusion pass): the du_dt/du_diss
        # split is keyed off ``config.ab2_scope`` inside the tendency kernel. To
        # run the split WITHOUT mutating the model's config, overlay just that one
        # field on a copy (a NamedTuple ``_replace`` — every other field
        # identical). ``None`` ⇒ pass the model config unchanged ⇒ bit-identical.
        _cfg = (_cfg_b._replace(ab2_scope=ab2_scope_override)
                if ab2_scope_override is not None else _cfg_b)
        return latlon_cgrid_ocean_baroclinic_tendencies(
            state, _grid, _zc, _cfg,
            physics_fn=self._physics_fn,
            surface_forcing=surface_forcing,
            sponge=sponge,
            dt=dt,
            surface_tracer_forcing_fn=self._surface_tracer_forcing_fn,
            vertex_mask=_vmask,
            momentum_only=momentum_only,
            skip_lateral_viscosity=skip_lateral_viscosity,
            precomputed_geom_density=precomputed_geom_density,
            ldf_state=ldf_state,
            zad_continuity_dt=zad_continuity_dt,
            zad_freshwater_eta_tendency=zad_freshwater_eta_tendency,
            momentum_flux_transport_velocity=momentum_flux_transport_velocity,
            up3_upwind_selector=up3_upwind_selector,
            momentum_flux_face_thickness=momentum_flux_face_thickness,
            ldf_thickness_operands=ldf_thickness_operands,
            ldf_metric_reciprocal_operands=ldf_metric_reciprocal_operands,
            ene_metric_reciprocals=ene_metric_reciprocals,
            ene_generic_f_vtx=ene_generic_f_vtx,
            legacy_hpg_algebraic=legacy_hpg_algebraic,
            nemo_operator_association=nemo_operator_association,
            nemo_stage_zad_operands=nemo_stage_zad_operands,
            nemo_stage_zad_operand_observer=nemo_stage_zad_operand_observer,
            nemo_stage_zad_eta_after_override=(
                nemo_stage_zad_eta_after_override),
            diagnose_momentum=return_nemo_operator_components,
            return_nemo_operator_components=return_nemo_operator_components,
        )

    def tendencies_with_diagnostics(
        self, state: LatLonCGridOceanState, surface_forcing=None,
        sponge=None, dt=300.0, *, grid=None, vertex_mask=None,
        ldf_state=None,
        z_coord=None, config=None,
        return_nemo_operator_components=False,
    ):
        """Compute baroclinic tendencies + per-term momentum-tendency
        breakdown.

        Returns
        -------
        (LatLonCGridOceanTendencies, MomentumTendencyDiagnostics)
            The diagnostics satisfy ``Σ components == du_dt`` to machine
            precision (verified by
            ``tests/ocean/unit/test_momentum_diagnostics_closure.py``)
            UNLESS ``config.adaptive_implicit_vertadv`` is set, in which
            case ``vertadv_{u,v}`` is a diagnostic-only start-of-step
            estimate excluded from ``du_dt`` and the closure becomes
            ``Σ (components except vertadv) == du_dt`` (see
            ``MomentumTendencyDiagnostics`` and
            ``latlon_cgrid_ocean_baroclinic_tendencies`` docstrings).

        Use the returned tendencies as the start-of-step approximation
        of what the model integrates internally; for the
        time-mean budget this converges to the actually-applied
        tendency at O(dt) accuracy.

        ``grid``/``vertex_mask`` (optional, SPMD): default ``None`` ->
        ``self.grid``/``self._vertex_mask`` (bit-identical); a band-local
        grid is injected by a future ``shard_map`` wrapper.

        ``ldf_state`` (T, S, u, v at the BEFORE level), forwarded verbatim to
        ``tendencies``, which has always accepted it.  IT MATTERS FOR ANY
        ORACLE COMPARISON: NEMO evaluates lateral friction at the before level
        (``dyn_ldf(kstp, Nbb, Nnn, ...)``, stpmlf.F90:319 -- a leapfrog-centred
        diffusion is unconditionally unstable, so this is required rather than
        incidental), and legoESM's production leapfrog path matches it.  This
        wrapper omitted the parameter, so every caller silently got NOW-level
        lateral friction while comparing against NEMO's BEFORE-level trend,
        measuring ``A_h*lap(u_now - u_before)`` -- a quantity the instrument
        manufactured.  Found 2026-08-25 by adversarial review of the
        zonal-wall momentum budget; the affected term was the only one in that
        budget whose difference was time-noisy.  Leave it ``None`` only for a
        non-leapfrog card or when the now/before distinction is genuinely
        irrelevant.
        """
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        _grid = grid if grid is not None else self.grid
        _vmask = vertex_mask if vertex_mask is not None else self._vertex_mask
        _qco_dt = dt
        if (getattr(_cfg_b, "zad_qco_evaluation", "generic") == "nemo_literal"
                and getattr(_cfg_b, "outer_integrator", "forward_euler")
                == "leapfrog"
                and getattr(state, "eta_before", None) is not None):
            # Public diagnostics receive the clock dt, while production MLF
            # calls _step_impl with rDt=2*dt.  Match that active source span.
            _qco_dt = 2.0 * dt
        return latlon_cgrid_ocean_baroclinic_tendencies(
            state, _grid, _zc, _cfg_b,
            physics_fn=self._physics_fn,
            surface_forcing=surface_forcing,
            sponge=sponge,
            dt=dt,
            diagnose_momentum=True,
            surface_tracer_forcing_fn=self._surface_tracer_forcing_fn,
            vertex_mask=_vmask,
            ldf_state=ldf_state,
            zad_continuity_dt=_qco_dt,
            return_nemo_operator_components=return_nemo_operator_components,
        )

    def _step_impl(self, state: LatLonCGridOceanState, dt: float,
                   freshwater=None, surface_forcing=None,
                   sponge=None, *, _apply_implicit_vmix: bool = True,
                   grid=None, vertex_mask=None, t_seconds=None,
                   _ab2_scope_override: str | None = None,
                   _barotropic_substep_scale: int = 1,
                   _barotropic_before_state=None,
                   _fct_tracer_before=None,
                   _external_tracer_rate=None,
                   _shortwave_tendency_test_delta=None,
                   _vertical_K_test_override=None,
                   _nemo_stage1_zad_eta_after_override=None,
                   _return_barotropic_substeps: bool = False,
                   _return_live_stage_operands: bool = False,
                   _return_tracer_process_trace: bool = False,
                   _return_ldf_diagnostic_trace: bool = False,
                   _ldf_state=None, _tke_n2_bundle_override=None,
                   _return_raw_kaa_qco: bool = False,
                   z_coord=None, config=None, iwm_fields=None):
        """Core step logic — no JIT wrapper.

        ``_external_tracer_rate`` (private, #1492 DINO ``surface_tendency_
        placement="leapfrog_rhs"``): optional ``(dT_dt, dS_dt)`` full-column
        rate pair [degC/s, PSU/s], already masked, SUMMED into the explicit
        ``tend.dT_dt``/``tend.dS_dt`` before the ``dt·`` combine below --
        i.e. it becomes part of the SAME RHS accumulator every other
        explicit tracer tendency uses, matching NEMO's ``tra_sbc`` writing
        into ``ts(Krhs)`` (trasbc.F90:150-155) BEFORE ``tra_zdf`` forms
        ``Kaa`` from it. ``None`` (every caller except the leap-frog Nnn
        pass under the new DINO placement) ⇒ bit-identical.

        ``_fct_tracer_before`` (private, MLF): ``(T_before, S_before)`` — the
        BEFORE-level (Nbb) tracers the leap-frog step passes so the FCT/Zalesak
        monotonicity base (low-order upwind flux, ``q_td``, ``q_min``/``q_max``
        bounds) is taken from Kbb, matching where the leap-frog APPLIES the
        limited advective increment (``T(Naa) = T(Nbb) + 2dt·RHS``).  The base
        is carried through the SAME pre-advection physics increment as the NOW
        tracer (``base = T_before + (T_mid − T_now)``), so the guaranteed-monotone
        after-state is the actual leap-frog after-state.  ``None`` (FE/AB2) ⇒
        base == NOW ⇒ byte-identical FCT (see ``fct_tracer_advection``).

        ``_barotropic_before_state`` (private, MLF): ``(eta_before, u_before,
        v_before)`` — the BEFORE-level (Nbb) ssh + 3-D velocity the leap-frog
        step (``_leapfrog_step``) passes so the split-explicit barotropic
        INTEGRATION is seeded from n-1 (NEMO ``ln_bt_fw=.FALSE.`` centred
        barotropic, ``dynspg_ts.F90:494-503``), while the frozen slow forcing
        stays at NOW.  Only the ``explicit_substep`` standard path consumes it;
        ``None`` (every other caller) seeds from the NOW state ⇒ bit-identical.

        ``_ldf_state`` (private, ``nemo_mlf`` P1 single-pass transcription):
        ``(T_ldf, S_ldf, u_ldf, v_ldf)`` — the BEFORE-level (Nbb) tracers +
        velocity that ``_nemo_mlf_step`` passes so ONLY the ``dyn_ldf``/
        ``tra_ldf`` lateral-friction/lateral-diffusion calls (``tendencies()``
        -> ``ldf_state``) AND the GM/Redi isoneutral-Redi tendency (row 28's
        lego home; ``stpmlf.F90:275``/``:437``) are evaluated on Nbb, while
        every other term in this SAME pass (advection, EEN vorticity, HPG,
        barotropic solve, GM bolus transport, physics, forcing) reads Nnn —
        replacing ``_leapfrog_step``'s two-``_step_impl``-pass mechanism for
        achieving the same Nbb-evaluated dissipation (see risk register #1
        item 4, ``nemo_mlf_step_transcription_spec.md`` §6).  ``None`` (every
        other caller, including ``_leapfrog_step``) ⇒ bit-identical.

        Use this directly inside an outer ``@jax.jit`` context (e.g.
        ``lax.scan`` block functions) to avoid nested JIT boundaries
        that can cause numerical divergence with partial-cell
        coordinates.  For standalone calls, use ``step()`` which wraps
        this in ``@jax.jit``.

        ``_apply_implicit_vmix`` (private): when ``False`` the final implicit
        vertical-mixing solve and the conservation fixer are skipped and the
        method returns ``(state_explicit, (tend.K_v, tend.A_v, K33))`` — the
        explicit-only forward-Euler state plus the implicit-mixing diffusivity
        profiles from the tendencies (``tend.K_v``/``tend.A_v`` may be ``None`` for
        schemes, e.g. TKE, that don't surface them — then the consumer recomputes,
        as the forward-Euler path does).  Used ONLY by the faithful AB2 path
        (``_ab2_step``), which AB2-extrapolates the explicit increment and then
        applies implicit vertical mixing ONCE (Veros core/thermodynamics.py +
        core/external/solve_stream.py).  The default ``True`` leaves every other
        caller bit-identical.

        ``grid``/``vertex_mask`` (optional, SPMD): when ``None`` (the
        production single-device path) the model's own ``self.grid`` /
        ``self._vertex_mask`` are used and behaviour is bit-identical to
        before; a future ``shard_map`` wrapper passes a per-device
        band-local grid (+ vertex mask) instead — ``_grid``/``_vmask`` are
        threaded into every tendency / barotropic / GM-Redi / EKE / TKE /
        conservation call this method makes that itself reads ``self.grid``.
        """
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        if self._nemo_ws_test_hooks.omit_barotropic_substep_drag:
            _cfg_b = _cfg_b._replace(barotropic_drag_substep=False)
        state = cast_pytree(state, None, "compute")
        # ``_ab2_scope_override`` (private): the leap-frog step (``_leapfrog_step``)
        # drives this method in "advective" scope to WITHHOLD the dissipative
        # tendencies from ``state_expl`` (so they land on the returned
        # ``diss_incr``) even though ``config.ab2_scope`` stays "total". NEMO
        # evaluates the explicit lateral diffusion / GM-Redi on the BEFORE level
        # (``dyn_ldf(Kbb)``/``tra_ldf(Kbb)``, forward-in-time), so the leap-frog
        # extracts ``diss_incr`` from a SEPARATE Nbb pass and combines it with the
        # advective Nnn increment. ``None`` (every other caller) ⇒ read the config
        # value ⇒ bit-identical. The override is threaded to ``self.tendencies``
        # (which performs the du_dt/du_diss split) AND the two ``_ab2_advective``
        # gates below.
        _scope = (_ab2_scope_override
                  if _ab2_scope_override is not None
                  else getattr(_cfg_b, "ab2_scope", "total"))
        # SPMD resolve: default None → the model's own grid / vertex mask
        # (bit-identical single-device path).
        _grid = grid if grid is not None else self.grid
        _vmask = vertex_mask if vertex_mask is not None else self._vertex_mask
        _tke_n2_bundle = _tke_n2_bundle_override
        _carried_slope_n2 = (
            _cfg_b.gm_redi is not None
            and getattr(
                _cfg_b.gm_redi, "slope_n2_evaluation", "recompute")
            == "carried_step_entry")
        if (_tke_n2_bundle is None
                and (_apply_implicit_vmix or _carried_slope_n2)):
            _tke_n2_bundle = self._tke_step_entry_n2_bundle(
                state, z_coord=_zc, config=_cfg_b)
        # Prescribed-flow lever (config.prescribed_flow, validated at
        # construction).  STATIC Python gate on the config value (CLAUDE.md
        # feature-gating exception): None (default) leaves every gated block
        # below untraced — the compiled graph is byte-identical to before the
        # lever existed.  Never jnp.where/lax.cond here.
        _pflow = getattr(_cfg_b, "prescribed_flow", None)

        # barotropic_forcing_centred (#1226 item 3; NEMO ln_bt_fw=.FALSE.,
        # dynspg_ts.F90:392-401 wind + :415-421 emp): rebind ``surface_forcing``
        # to the ½(before+now) TAU average BEFORE it reaches
        # ``self.tendencies()`` — lego deposits wind stress as a SINGLE explicit
        # top-cell kick (``_bc_external_surface_forcing``, always-on) that plays
        # BOTH of NEMO's separately-centred wind consumers at once (the 2D
        # zu_frc barotropic RHS AND the dynzdf top-cell implicit-solve BC, which
        # under MLF both average utau_b+utauU) — so centring the single lego
        # deposit reproduces both NEMO terms together. Freshwater is NOT
        # rebound here (only the eta/ssh_frc channel is centred, done
        # surgically at the F_slow_eta call site below; the virtual-salt-flux
        # tracer deposit stays at NOW). NEMO nit000 seeding (sbcmod.F90:
        # 568-573): ``state.tau_x_prev`` is None on the very first leapfrog
        # step (before ``u_before`` exists) -- that step takes the
        # forward-Euler branch in ``_leapfrog_step`` before this function
        # ever runs with centring on, so no seeding branch is needed here;
        # every step this function runs under centring has a real
        # ``tau_x_prev`` (seeded to the pre-step NOW value by
        # ``_leapfrog_step``, matching NEMO's "before := now" nit000 rule).
        if (getattr(_cfg_b, "barotropic_forcing_centred", False)
                and surface_forcing is not None
                and getattr(surface_forcing, "tau_x", None) is not None
                and getattr(surface_forcing, "tau_y", None) is not None
                and getattr(state, "tau_x_prev", None) is not None
                and getattr(state, "tau_y_prev", None) is not None):
            surface_forcing = surface_forcing._replace(
                tau_x=0.5 * (state.tau_x_prev + surface_forcing.tau_x),
                tau_y=0.5 * (state.tau_y_prev + surface_forcing.tau_y),
            )

        # Asynchronous ("distorted-physics") time stepping: the public ``dt`` IS
        # dt_tracer (the clock; Veros advances vs.time by dt_tracer). Momentum + the
        # barotropic solve + implicit vertical FRICTION use the shorter dt_mom; the
        # tracer/continuity/eta/EKE/freshwater path + implicit vertical DIFFUSION keep
        # ``dt`` (=dt_tracer). dt_mom_ratio=1.0 (default) ⇒ dt_mom == dt ⇒ bit-identical.
        # Validation requires barotropic_solver="rigid_lid" when ratio != 1.0 (fixed
        # column depth ⇒ exact tracer conservation; Veros's streamfunction rigid lid).
        dt_mom = dt / _cfg_b.dt_mom_ratio

        u_mask_3d = state.u_mask.data[..., jnp.newaxis]
        v_mask_3d = state.v_mask.data[..., jnp.newaxis]
        mask_3d = state.land_mask.data[..., jnp.newaxis]

        # 1. Baroclinic tendencies (non-Coriolis)
        # The geometry + density (EOS) + baroclinic pressure anomaly depend
        # ONLY on eta/T/S, which are FROZEN across this step's stage-1 tendency
        # AND the RK3 momentum sub-stages -> compute the bundle ONCE and reuse
        # it everywhere (#25), so _bc_geometry_and_density (the EOS +
        # pressure-anomaly recompute) runs 1x/step instead of 3x under RK3.
        # Bit-identical (compute_frozen_geom_density mirrors the in-fn stages
        # 1-3 derivation on the same frozen state; pinned by the parity gate).
        _geom_density = compute_frozen_geom_density(
            state, _grid, _zc, _cfg_b)
        # The UP3 T-point upwind selector is keyed by the REFERENCE the config's
        # ``momentum_flux_scheme`` names ("nemo_up3" -> the advected-velocity
        # pair per dynadv_up3.F90:166-170; "oceananigans_up3" -> the transport
        # pair), NOT by the time integrator -- see UP3_REFERENCE_SELECTOR.
        # ``None`` therefore means "use the scheme's own reference rule" at
        # every stage (stage 1 here, stages 2-3 in _mom_pert_ws).  The private
        # hook forces the legacy transport sign for the stage-sweep gate's
        # one-variable ablation arm only; no card sets it.
        _up3_selector_override = (
            "transport"
            if self._nemo_ws_test_hooks.legacy_up3_transport_sign_selector
            else None)
        # The momentum-advection FACE THICKNESS, unlike the selector above,
        # IS a property of the WS-RK3 stage program (isomorphism row S-47):
        # NEMO's dyn_adv_up3 consumes e3u(Kmm) while legoESM's tendencies()
        # rebuilt its own min-of-stretched-T pair.  ``None`` (every other
        # integrator) keeps that historical min rule.
        _ws_face_thickness_kbb = None
        _ws_ldf_face_thickness_kbb = None
        _ws_ldf_thickness_kbb = None
        _ws_uses_nemo_ldf_e3 = (
            _cfg_b.lateral_viscosity_operator == "nemo_div_curl"
            and _cfg_b.lateral_viscosity_e3_weighting == "nemo_e3")
        if getattr(_cfg_b, "momentum_time_integrator", "euler") == "rk3_ws":
            # NEMO's e3u/e3v(Kbb) = e3u_0*(1+r3u(Kbb)) for the step-entry
            # dyn_adv (stp2d.F90:172 -> dynadv_up3.F90:160; the same pair
            # stage 1 consumes through zFu at stprk3_stg.F90:273-274), from
            # the ONE shared kernel; ``None`` (the private legacy hook) keeps
            # tendencies()' own min-of-stretched-thicknesses rule.
            if isinstance(_zc, OceanPartialCellCoordinate):
                _ws_u_live_mask, _ws_v_live_mask = compute_face_masks_3d(
                    _zc.is_active, _grid)
                _ws_u_live_mask = _ws_u_live_mask.astype(state.eta.data.dtype)
                _ws_v_live_mask = _ws_v_live_mask.astype(state.eta.data.dtype)
            else:
                # An OceanZStarCoordinate carries no per-level active flag, so
                # the 2-D face mask IS the only mask available on that path;
                # a staircase there keeps the pre-fix behaviour.  Recorded,
                # not silently guarded: no certified card takes this branch
                # (both testcase cards build OceanPartialCellCoordinate).
                _ws_u_live_mask = state.u_mask.data[..., None]
                _ws_v_live_mask = state.v_mask.data[..., None]
            # stprk3_stg.F90:367 (vec/linssh), :375 (the compiled key_qco
            # branch) and :382 (the #else) multiply the stage velocity by
            # ``umask(ji,jj,jk)``; :444 adds the barotropic correction as
            # ``zub(ji,jj)*umask(ji,jj,jk)``; and :273 masks the SAME
            # correction inside the advective transport,
            # ``zFu = e2u*e3u(Kmm)*( uu(Kmm) + zub*umask(ji,jj,jk) )``.
            # NEMO's velocity and its advective transport are therefore
            # EXACTLY zero below the seabed, which is what dyn_adv_up3's
            # k-slab stencil reads when it reaches a dry face
            # (dynadv_up3.F90:142-143 ``zlu_uu``, :160 ``zFu``, :166-176
            # ``zFu_t``): it does not skip a dry neighbour, it reads its zero.
            # Masking with the 2-D face mask broadcast over levels left the
            # depth-mean increment standing at every level below a staircase
            # face's own seabed, where the deeper neighbour's wet bottom level
            # then read it as a stencil neighbour.  The private hook restores
            # the 2-D rule; NEMO has no such switch.
            _ws_stage_u_mask = (
                state.u_mask.data[..., jnp.newaxis]
                if self._nemo_ws_test_hooks.legacy_2d_stage_face_mask
                else _ws_u_live_mask)
            _ws_stage_v_mask = (
                state.v_mask.data[..., jnp.newaxis]
                if self._nemo_ws_test_hooks.legacy_2d_stage_face_mask
                else _ws_v_live_mask)
            _ws_h_ref = compute_layer_thickness(
                jnp.zeros_like(state.eta.data), state.H_bathy.data, _zc,
                min_water_column_m=_cfg_b.min_water_column_m)
            if not self._nemo_ws_test_hooks.legacy_hadv_min_face_thickness:
                _ws_face_thickness_kbb = _nemo_ws_qco_stage_faces(
                    state.eta.data, _ws_h_ref, _ws_u_live_mask,
                    _ws_v_live_mask, _grid)[:2]
            if _ws_uses_nemo_ldf_e3:
                _ws_ldf_face_thickness_kbb = (
                    _ws_face_thickness_kbb
                    if _ws_face_thickness_kbb is not None
                    else _nemo_ws_qco_stage_faces(
                        state.eta.data, _ws_h_ref, _ws_u_live_mask,
                        _ws_v_live_mask, _grid)[:2])
                from legoesm.ocean.vertical import (
                    nemo_ldf_reference_e3f,
                    nemo_qco_live_vorticity_e3f_cgrid,
                )
                _ws_t_live_mask = getattr(_zc, "is_active", None)
                if _ws_t_live_mask is None:
                    _ws_t_live_mask = jnp.broadcast_to(
                        state.land_mask.data[..., None], _ws_h_ref.shape)
                # dynldf_lev.f90:123 stretches e3f_3d, the MESH reference F
                # thickness; only dynvor.f90:734-738 stretches its own frozen
                # e3f_0vor.  The two consumers share r3f and fe3mask, not the
                # reference, so lateral diffusion asks for its own.
                _ws_e3f_kbb = nemo_qco_live_vorticity_e3f_cgrid(
                    state.eta.data, _zc, state.eta.data.dtype, grid=_grid,
                    e3t_0=_ws_h_ref, tmask=_ws_t_live_mask,
                    reference_e3f=nemo_ldf_reference_e3f(_zc))
                _ws_ldf_thickness_kbb = (
                    _geom_density[1], _ws_ldf_face_thickness_kbb[0],
                    _ws_ldf_face_thickness_kbb[1], _ws_e3f_kbb,
                    _ws_ldf_face_thickness_kbb[0],
                    _ws_ldf_face_thickness_kbb[1])
        # The per-term observer needs the diagnostics this same call can
        # already return; asking for them adds the decomposition and changes
        # no tendency (the diagnostics are built from the terms as they are
        # accumulated).  The live-stage operand bundle stays gated on its own
        # flag so a per-term measurement cannot switch a production arm on.
        _rhs_term_observer = (
            self._nemo_ws_test_hooks.slow_forcing_rhs_term_observer)
        _want_rhs_components = (
            _return_live_stage_operands or callable(_rhs_term_observer)
            or bool(self._nemo_ws_test_hooks
                    .expose_stage1_momentum_rhs_split))
        _tend_result = self.tendencies(
                               state, surface_forcing, sponge=sponge, dt=dt,
                               precomputed_geom_density=_geom_density,
                               grid=_grid, vertex_mask=_vmask,
                               ab2_scope_override=_ab2_scope_override,
                               ldf_state=_ldf_state, z_coord=z_coord, config=config,
                               zad_continuity_dt=dt,
                               up3_upwind_selector=_up3_selector_override,
                               momentum_flux_face_thickness=_ws_face_thickness_kbb,
                               ldf_thickness_operands=_ws_ldf_thickness_kbb,
                               nemo_operator_association=self._nemo_ws_test_hooks.nemo_stage_rhs_accumulation_order_arm,
                               nemo_stage_zad_operands=(
                                   (self._nemo_ws_test_hooks.stage1_zad_w_override, None, None)
                                   if self._nemo_ws_test_hooks.stage1_zad_w_override is not None else None),
                               nemo_stage_zad_operand_observer=(
                                   self._nemo_ws_test_hooks.stage1_zad_operand_observer),
                               nemo_stage_zad_eta_after_override=(
                                   _nemo_stage1_zad_eta_after_override),
                               return_nemo_operator_components=_want_rhs_components)
        if _want_rhs_components:
            tend, _mom_term_diagnostics, _live_operands = _tend_result
            _nemo_ws_stage1_operator_operands = (
                _live_operands if _return_live_stage_operands else None)
            # dyn_adv's content as THIS evaluation accumulated it: the half
            # of the step-level right-hand side that NEMO's three-dimensional
            # pre-stage array does not carry.  ``advection_u`` is
            # -dKE_dx + Dterm + vertadv and ``flux_form_hadv_u`` the
            # flux-form horizontal trend, which shares its diagnostic slot
            # with the rotation terms and is therefore published apart
            # (zero on every vector-invariant card, where the KE gradient
            # inside ``advection_u`` carries the horizontal half instead).
            # Carried APART as well as summed: on a flux-form card
            # ``advection_u`` is the vertical UP3 term alone (the kinetic
            # energy gradient is zeroed there and the WENO D-term is
            # inactive) and ``flux_form_hadv_u`` is the horizontal flux
            # divergence, which are the two halves ``dyn_adv_up3`` writes
            # (dynadv_up3.f90:174-215 and :245-360).
            _nemo_ws_stage1_advection_halves = (
                (_live_operands["flux_form_hadv_u"].data,
                 _live_operands["flux_form_hadv_v"].data),
                (_live_operands["advection_u"].data,
                 _live_operands["advection_v"].data))
            _nemo_ws_stage1_main_advection = (
                _live_operands["advection_u"].data
                + _live_operands["flux_form_hadv_u"].data,
                _live_operands["advection_v"].data
                + _live_operands["flux_form_hadv_v"].data)
            if callable(_rhs_term_observer):
                jax.debug.callback(
                    _rhs_term_observer,
                    {name: getattr(_mom_term_diagnostics, name).data
                     for name in type(_mom_term_diagnostics)._fields},
                    ordered=False)
        else:
            tend = _tend_result
            _nemo_ws_stage1_operator_operands = None
            _nemo_ws_stage1_main_advection = None
            _nemo_ws_stage1_advection_halves = None
        # #1492 DINO surface_tendency_placement="leapfrog_rhs": fold the
        # externally-supplied surface tracer RATE into the SAME explicit RHS
        # every other tendency uses -- BEFORE the diss-withholding split and
        # the dt-combine below, matching NEMO's tra_sbc(Nnn, ts, Nrhs) writing
        # into the shared Krhs accumulator ahead of tra_zdf's Kaa combine.
        # None (default / every caller except the leap-frog Nnn pass under
        # the new placement) ⇒ bit-identical.
        if _external_tracer_rate is not None:
            _dT_ext, _dS_ext = _external_tracer_rate
            tend = tend._replace(
                dT_dt=tend.dT_dt.replace(data=tend.dT_dt.data + _dT_ext),
                dS_dt=tend.dS_dt.replace(data=tend.dS_dt.data + _dS_ext),
            )
        # GYRE causal-arm controls.  Each changes exactly one diagnosed
        # operand and is reachable only through this private implementation
        # method; production ``step`` never supplies either argument.
        if _shortwave_tendency_test_delta is not None:
            tend = tend._replace(dT_dt=tend.dT_dt.replace(
                data=tend.dT_dt.data + _shortwave_tendency_test_delta))
        # AB2 "advective" scope (Veros-faithful): the DISSIPATIVE tendencies are
        # WITHHELD from tend.{du,dv,dT,dS}_dt and exposed on tend.{...}_diss so
        # they can be applied at WEIGHT 1.0 (forward-Euler) rather than being
        # AB2-extrapolated. Accumulate the weight-1.0 dissipative INCREMENT here
        # (dt · dissipative-rate, evaluated on the PRE-STEP state u^n/T^n — like
        # Veros's du_mix / hor_diffusion on ``tr[tau]``); the GM/Redi
        # isoneutral+skew tracer dissipation is added below (where it is
        # computed). ``_ab2_step`` adds this increment at weight 1.0; the
        # forward-Euler step() path applies it inline so a non-AB2 outer
        # integrator stays faithful too. ``None`` carry under "total" ⇒
        # bit-identical (the diss fields are None ⇒ this whole block is skipped).
        _ab2_advective = (_scope == "advective")
        _diss_dT_incr = None
        _diss_dS_incr = None
        _diss_du_incr = None
        _diss_dv_incr = None
        if _ab2_advective and tend.dT_diss is not None:
            _diss_dT_incr = dt * tend.dT_diss.data
            _diss_dS_incr = dt * tend.dS_diss.data
            _diss_du_incr = dt_mom * tend.du_diss.data
            _diss_dv_incr = dt_mom * tend.dv_diss.data

        # 2. Update tracers
        T_new = state.T.data + dt * tend.dT_dt.data
        S_new = state.S.data + dt * tend.dS_dt.data

        # 3. Slow-forcing coupling (#205): split baroclinic tendency into
        # depth-averaged (slow forcing for barotropic solver) and
        # perturbation (applied to 3D velocity before barotropic step).
        #
        # Current approach (without slow-forcing): apply full tendency
        # to u_star, then barotropic solver sees initial U_bar that
        # already includes the depth-averaged tendency.  The problem:
        # the barotropic solver doesn't know about this forcing, so
        # the baroclinic-barotropic coupling is only through the initial
        # velocity — not updated as eta evolves during substeps.
        #
        # MOM6 approach: pass depth-averaged tendency as F_slow_u to the
        # barotropic solver, which applies it at each substep.  This
        # couples the slow forcing to the evolving barotropic state.
        du_dt = tend.du_dt.data
        dv_dt = tend.dv_dt.data
        _slow_rhs_override = self._nemo_ws_test_hooks.slow_forcing_rhs_override
        if _slow_rhs_override is not None:
            _slow_rhs_u, _slow_rhs_v = _slow_rhs_override
            du_dt = du_dt.at[:, 1:, :].set(_slow_rhs_u)
            dv_dt = dv_dt.at[1:, :, :].set(_slow_rhs_v)
        _slow_rhs_observer = self._nemo_ws_test_hooks.slow_forcing_rhs_observer
        if callable(_slow_rhs_observer):
            if self._nemo_ws_test_hooks.slow_forcing_rhs_observer_face == "u":
                jax.debug.callback(_slow_rhs_observer, du_dt, ordered=False)
            else:
                jax.debug.callback(_slow_rhs_observer, dv_dt, ordered=False)

        # Compute layer thickness at u/v faces for depth-averaging.
        # Min-rule: the face's effective wet thickness is the shallower
        # side's thickness (MOM6/MITgcm hFacW convention).  The same
        # convention is used in the barotropic Helmholtz solver and the
        # tracer mass flux below — consistency is required for the
        # Hallberg-Adcroft 2009 column-sum invariant
        # ``sum_k(h_u * u_corrected) == Hu_avg`` to hold to machine
        # precision.  For full cells this reduces to the cell value
        # (bit-exact backwards-compat).
        h_k_pre = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, _zc,
            min_water_column_m=_cfg_b.min_water_column_m,
        )
        # h at u-faces — min-rule (MOM6/MITgcm hFacW convention).
        # Must match the PE tendency which uses min_cell_to_uface, so
        # that F_slow = depth_avg(du_dt, h_u) is consistent with the
        # 3D tendency.  Arithmetic mean overestimates face depth at
        # topographic steps, creating a barotropic-baroclinic residual.
        h_u_pre = min_cell_to_uface(h_k_pre)
        # h at v-faces — same min-rule for meridional direction.
        h_v_pre = min_cell_to_vface(h_k_pre, _grid)

        # H + F_slow share the per-face h weight on the level axis —
        # fuse the two reductions per face into one stacked sum.
        _u_pair = jnp.sum(jnp.stack([h_u_pre, du_dt * h_u_pre], axis=-1), axis=-2)
        H_u_pre = jnp.maximum(_u_pair[..., 0], 1e-10)
        F_slow_u = _u_pair[..., 1] / H_u_pre * state.u_mask.data
        _v_pair = jnp.sum(jnp.stack([h_v_pre, dv_dt * h_v_pre], axis=-1), axis=-2)
        H_v_pre = jnp.maximum(_v_pair[..., 0], 1e-10)
        F_slow_v = _v_pair[..., 1] / H_v_pre * state.v_mask.data
        # NEMO's own depth average of the slow forcing (stp2d.F90:177-186 of
        # the VORTEX_SMT builds): the REFERENCE face thickness and the STORED
        # reciprocal, with no ssh stretching anywhere in the statement --
        #   Ue_rhs = SUM( e3u_0(1:jpkm1)*uu(Krhs)*umask ) * r1_hu_0
        # where e3u_0 is the min of the two neighbouring REFERENCE
        # thicknesses and r1_hu_0 = ssumask/(hu_0 + 1 - ssumask) with
        # hu_0 = SUM_k e3u_0*umask (domain.F90).  The min-rule-live form
        # above equals this on a FULL-STEP mesh (one per-face scalar that
        # cancels) and does NOT over partial cells, where the per-level
        # minimum of the two STRETCHED thicknesses can follow a different
        # column than the reference minimum does.  Static Python branch on a
        # config string, so the default path is bit-identical.
        # DECISION 90 (user): no default.  An unset field is a card that
        # never stated which depth average it runs, which is the hidden
        # choice this field exists to remove -- so it raises rather than
        # silently selecting one.
        #
        # SCOPE OF THE RAISE, stated rather than left implicit.  The choice
        # is between NEMO's stp2d.F90:177-186 statement and legoESM's own
        # live min-rule, so it only EXISTS on a card running NEMO's RK3
        # momentum program; a card on legoESM's own time stepping executes
        # no NEMO statement here and has nothing to choose between.  This
        # is the same shape as the nemo_first_wzv_after_ssh guard above,
        # which refuses a NEMO value on a non-RK3 program instead of
        # demanding one from every card.
        _slow_depth_eval = getattr(
            _cfg_b.barotropic, "barotropic_slow_forcing_depth_evaluation", "")
        _nemo_rk3_family = getattr(
            _cfg_b, "momentum_time_integrator", "euler") in ("rk3", "rk3_ws")
        if not _slow_depth_eval:
            if _nemo_rk3_family:
                raise ValueError(
                    "barotropic_slow_forcing_depth_evaluation is unset: this "
                    "card runs NEMO's RK3 momentum program and must STATE "
                    "how the slow forcing is depth-averaged onto the "
                    "barotropic faces -- 'nemo_literal' for NEMO's own "
                    "statement (stp2d.F90:177-186) or 'min_rule_live' for "
                    "the per-level minimum of the two live thicknesses. "
                    "There is no default (decision 90).")
            _slow_depth_eval = "min_rule_live"
        if _slow_depth_eval not in ("min_rule_live", "nemo_literal"):
            raise ValueError(
                "unknown barotropic_slow_forcing_depth_evaluation scheme "
                f"{_slow_depth_eval!r}: must be one of "
                "('min_rule_live', 'nemo_literal').")
        if _slow_depth_eval == "nemo_literal":
            from legoesm.ocean.vertical import nemo_qco_card_mesh_operands
            _h_ref0 = compute_layer_thickness(
                jnp.zeros_like(state.eta.data), state.H_bathy.data, _zc,
                min_water_column_m=_cfg_b.min_water_column_m,
            ).astype(du_dt.dtype)
            if isinstance(_zc, OceanPartialCellCoordinate):
                _um3, _vm3 = compute_face_masks_3d(_zc.is_active, _grid)
                _um3 = _um3.astype(du_dt.dtype)
                _vm3 = _vm3.astype(du_dt.dtype)
            else:
                _um3 = jnp.asarray(u_mask_3d, dtype=du_dt.dtype)
                _vm3 = jnp.asarray(v_mask_3d, dtype=du_dt.dtype)
            _ops0 = nemo_qco_card_mesh_operands(
                _h_ref0, _um3, _vm3, _grid, du_dt.dtype)
            _one = jnp.asarray(1.0, dtype=du_dt.dtype)
            _wet_u0 = (_ops0.hu_0 > 0.0).astype(du_dt.dtype)
            _wet_v0 = (_ops0.hv_0 > 0.0).astype(du_dt.dtype)
            _r1_hu_0 = _wet_u0 / (_ops0.hu_0 + _one - _wet_u0)
            _r1_hv_0 = _wet_v0 / (_ops0.hv_0 + _one - _wet_v0)
            _slow_u_native = jnp.sum(
                _ops0.e3u_0 * du_dt[:, 1:, :] * _ops0.umask3,
                axis=-1) * _r1_hu_0
            _slow_v_native = jnp.sum(
                _ops0.e3v_0 * dv_dt[1:, :, :] * _ops0.vmask3,
                axis=-1) * _r1_hv_0
            F_slow_u = (F_slow_u.at[:, 1:].set(_slow_u_native)
                        * state.u_mask.data)
            F_slow_v = (F_slow_v.at[1:, :].set(_slow_v_native)
                        * state.v_mask.data)
        _slow_depth_override = (
            self._nemo_ws_test_hooks.slow_forcing_depth_override)
        if _slow_depth_override is not None:
            _slow_depth_u, _slow_depth_v = _slow_depth_override
            F_slow_u = F_slow_u.at[:, 1:].set(_slow_depth_u)
            F_slow_v = F_slow_v.at[1:, :].set(_slow_depth_v)
        _F_slow_depth_u = F_slow_u
        _F_slow_depth_v = F_slow_v

        _wind_tau_i_u = jnp.zeros_like(F_slow_u)
        _wind_tau_j_v = jnp.zeros_like(F_slow_v)
        _wind_r1_rho0 = jnp.asarray(0.0, dtype=du_dt.dtype)
        _wind_r1_hu = jnp.zeros_like(F_slow_u)
        _wind_r1_hv = jnp.zeros_like(F_slow_v)
        _wind_increment_u = jnp.zeros_like(F_slow_u)
        _wind_increment_v = jnp.zeros_like(F_slow_v)

        if getattr(_cfg_b, "surface_stress_implicit", False):
            # NEMO stp2d explicit barotropic wind term: with the stress
            # WITHHELD from du_dt (surface_stress_implicit; deposited inside
            # the implicit vertical solve instead), F_slow no longer carries
            # its depth mean — add tau/(rho0 H) here so the barotropic mode
            # keeps the wind forcing (sign: ocean-reaction, same helper as
            # the deposition; zero when the forcing carries no stress).
            from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
                surface_stress_faces,
            )
            from legoesm.ocean.vertical import compute_ocean_jacobian
            _J_fs = compute_ocean_jacobian(
                state.eta.data, state.H_bathy.data, _zc)
            _sfx = (surface_stress_faces(
                        surface_forcing, du_dt.dtype, _zc, _J_fs,
                        _grid,
                        use_native=not (
                            self._nemo_ws_test_hooks
                            .legacy_geographic_surface_stress_arm),
                        **({} if self._nemo_ws_test_hooks
                            .legacy_coastal_surface_stress_factors else {
                                "cell_mask": state.land_mask.data,
                                "u_mask": state.u_mask.data,
                                "v_mask": state.v_mask.data,
                            }))
                    if surface_forcing is not None else None)
            if _sfx is not None:
                _tau_i_u, _tau_j_v, _, _ = _sfx
                _r0 = jnp.asarray(_cfg_b.constants.rho_0,
                                  dtype=du_dt.dtype)
                _wind_tau_i_u = _tau_i_u
                _wind_tau_j_v = _tau_j_v
                _wind_r1_rho0 = nemo_source_round(1.0 / _r0)
                _wind_r1_hu = nemo_source_round(1.0 / H_u_pre)
                _wind_r1_hv = nemo_source_round(1.0 / H_v_pre)
                _wind_operand_override = (
                    self._nemo_ws_test_hooks
                    .slow_forcing_wind_operand_override)
                if _wind_operand_override is not None:
                    (_rho_override, _stress_override,
                     _inverse_depth_override) = _wind_operand_override
                    if _rho_override is not None:
                        _wind_r1_rho0 = jnp.asarray(
                            _rho_override, dtype=du_dt.dtype)
                    if _stress_override is not None:
                        _stress_u, _stress_v = _stress_override
                        _wind_tau_i_u = _wind_tau_i_u.at[:, 1:].set(_stress_u)
                        _wind_tau_j_v = _wind_tau_j_v.at[1:, :].set(_stress_v)
                    if _inverse_depth_override is not None:
                        _r1_hu_override, _r1_hv_override = (
                            _inverse_depth_override)
                        _wind_r1_hu = _wind_r1_hu.at[:, 1:].set(
                            _r1_hu_override)
                        _wind_r1_hv = _wind_r1_hv.at[1:, :].set(
                            _r1_hv_override)
                if not self._nemo_ws_test_hooks.legacy_barotropic_wind_association:
                    # stp2d.F90:200-201, preserving Fortran's written
                    # ``(r1_rho0*tau)*r1_h`` product before the addition.
                    _wind_increment_u = nemo_source_round(
                        nemo_source_round(_wind_r1_rho0 * _wind_tau_i_u)
                        * _wind_r1_hu)
                    _wind_increment_v = nemo_source_round(
                        nemo_source_round(_wind_r1_rho0 * _wind_tau_j_v)
                        * _wind_r1_hv)
                    F_slow_u = nemo_source_round(
                        F_slow_u + _wind_increment_u) * state.u_mask.data
                    F_slow_v = nemo_source_round(
                        F_slow_v + _wind_increment_v) * state.v_mask.data
                else:
                    _wind_increment_u = _tau_i_u / (_r0 * H_u_pre) \
                        * state.u_mask.data
                    _wind_increment_v = _tau_j_v / (_r0 * H_v_pre) \
                        * state.v_mask.data
                    F_slow_u = F_slow_u + _wind_increment_u
                    F_slow_v = F_slow_v + _wind_increment_v
        _F_slow_wind_u = F_slow_u
        _F_slow_wind_v = F_slow_v

        # Perturbation tendency (depth-mean removed) → applied to 3D.
        # MUST be computed from the *baroclinic-only* F_slow (before A2 is
        # added below) so that the depth-mean biharmonic damping acts only
        # on U_bar, not on the perturbation u' = u - U_bar.
        du_dt_pert = du_dt - F_slow_u[..., jnp.newaxis]
        dv_dt_pert = dv_dt - F_slow_v[..., jnp.newaxis]

        # NEMO dyn_drg_init baroclinic-residual drag correction (#1226;
        # dynspg_ts.F90:1614-1643, DINO branch ln_isfcav=F/ln_drgice_imp=F):
        #   pCdU_u  = r1_2*(rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj))        (:1616)
        #   zu_i    = puu(ji,jj,ikbu,Kmm) - puu_b(ji,jj,Kmm)          (:1627)
        #   pu_RHSi += r1_hu(ji,jj,Kmm) * pCdU_u * zu_i               (:1642)
        # SIGN WALK (positive-r convention here): NEMO rCdU_bot <= 0 ⇒
        # pCdU_u = -r_eff (r_eff >= 0, the shared nemo_bottom_drag_rate_faces
        # 0.5-average), so the term is  -r_eff/H_u · (u_bot − U_bar): it
        # DAMPS the bottom-cell baroclinic residual's projection onto the
        # barotropic RHS (F_slow = NEMO's zu_frc).  Placed AFTER the
        # du_dt_pert split above because NEMO removes the vertical mean from
        # puu(Krhs) at :344-347 BEFORE dyn_drg_init runs — the correction
        # lives ONLY in the barotropic forcing, never in the 3-D residual.
        # Time level: NOW (:1627, ln_bt_fw=T form) by default.  DINO's
        # namelist runs ln_bt_fw=F (CENTRED → Kbb residual, :1634 — NOT a
        # before/now AVERAGE like the wind/emp terms above; the ln_bt_fw=F
        # branch reads puu(Kbb)/puu_b(Kbb) OUTRIGHT, pure BEFORE). Under
        # ``barotropic_forcing_centred=True`` the velocity source switches to
        # ``state.u_before``/``v_before`` (Kbb) to match; the drag RATE +
        # thickness scaling stay at NOW (h_k_pre/H_u_pre, matching NEMO's
        # r1_hu(Kmm) at :1642 — only the velocity residual is Kbb-gated, not
        # the geometry/rate). Default False -> the NOW form (bit-identical).
        if getattr(_cfg_b, "barotropic_drag_substep", False):
            from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
                nemo_bottom_drag_rate_faces,
            )
            _centred_drag = (
                getattr(_cfg_b, "barotropic_forcing_centred", False)
                and getattr(state, "u_before", None) is not None
                and getattr(state, "v_before", None) is not None)
            _u_src = state.u_before.data if _centred_drag else state.u.data
            _v_src = state.v_before.data if _centred_drag else state.v.data
            _r_u_bt, _r_v_bt, _isb_u, _isb_v = nemo_bottom_drag_rate_faces(
                state.u.data, state.v.data, h_k_pre, _zc,
                _cfg_b, _grid)
            _u_bot = jnp.sum(_u_src * _isb_u, axis=-1)
            _v_bot = jnp.sum(_v_src * _isb_v, axis=-1)
            _U_bar_now = jnp.sum(_u_src * h_u_pre, axis=-1) / H_u_pre
            _V_bar_now = jnp.sum(_v_src * h_v_pre, axis=-1) / H_v_pre
            F_slow_u = (F_slow_u
                        - _r_u_bt.astype(F_slow_u.dtype) / H_u_pre
                        * (_u_bot - _U_bar_now)) * state.u_mask.data
            F_slow_v = (F_slow_v
                        - _r_v_bt.astype(F_slow_v.dtype) / H_v_pre
                        * (_v_bot - _V_bar_now)) * state.v_mask.data
        _slow_drag_override = (
            self._nemo_ws_test_hooks.slow_forcing_drag_override)
        if _slow_drag_override is not None:
            _slow_drag_u, _slow_drag_v = _slow_drag_override
            F_slow_u = F_slow_u.at[:, 1:].set(_slow_drag_u)
            F_slow_v = F_slow_v.at[1:, :].set(_slow_drag_v)
            _F_slow_drag_u = F_slow_u
            _F_slow_drag_v = F_slow_v
            F_slow_u = nemo_source_round(
                F_slow_u + _wind_increment_u) * state.u_mask.data
            F_slow_v = nemo_source_round(
                F_slow_v + _wind_increment_v) * state.v_mask.data
            _F_slow_wind_u = F_slow_u
            _F_slow_wind_v = F_slow_v
        else:
            _F_slow_drag_u = F_slow_u
            _F_slow_drag_v = F_slow_v

        # A2 — depth-mean biharmonic hyperviscosity on (U_bar, V_bar).
        # Damps the barotropic standing mode at deep cells next to steep
        # slopes (Rhines 1969 bottom-trapped wave with f≈0) without
        # touching the baroclinic perturbation u' (already finalized
        # above as du_dt_pert / dv_dt_pert).  Applied as an additional
        # slow forcing on the implicit-CN barotropic solver:
        #   ∂U_bar/∂t |_diss = -ν₄ · ∇⁴ U_bar.
        # MOM6/HIM BIHARMONIC_BAROTROPIC analog.  No-op at default
        # ``B_h_barotropic = 0`` (bit-exact backward compat).
        if getattr(_cfg_b.lateral_viscosity, "B_h_barotropic", 0.0) > 0.0:
            from legoesm.ocean.dynamics.latlon_cgrid_operators import (
                vector_bilaplacian_cgrid, biharmonic_scaling_factor,
            )
            U_bar = jnp.sum(state.u.data * h_u_pre, axis=-1) / H_u_pre
            V_bar = jnp.sum(state.v.data * h_v_pre, axis=-1) / H_v_pre
            U_bar = U_bar * state.u_mask.data
            V_bar = V_bar * state.v_mask.data
            bilap_U, bilap_V = vector_bilaplacian_cgrid(
                U_bar, V_bar, _grid,
                mask=state.land_mask.data,
                u_mask=state.u_mask.data,
                v_mask=state.v_mask.data,
                vertex_mask=_vmask,
            )
            # cos^4(lat) scaling: lat-lon grid spacing shrinks as
            # cos(lat) near the poles, so a constant ν₄ would violate
            # biharmonic CFL there.  Same convention as the layered B_h.
            scale_u, scale_v = biharmonic_scaling_factor(_grid)
            nu4 = jnp.asarray(
                _cfg_b.lateral_viscosity.B_h_barotropic, dtype=F_slow_u.dtype,
            )
            scale_u_b = scale_u.astype(F_slow_u.dtype)[:, None]
            scale_v_b = scale_v.astype(F_slow_v.dtype)[:, None]
            F_slow_u = F_slow_u - nu4 * scale_u_b * bilap_U.astype(F_slow_u.dtype)
            F_slow_v = F_slow_v - nu4 * scale_v_b * bilap_V.astype(F_slow_v.dtype)
            F_slow_u = F_slow_u * state.u_mask.data
            F_slow_v = F_slow_v * state.v_mask.data

        # AB2 time-centering of the barotropic slow forcing (matches the
        # Oceananigans split-explicit Gᵁ = AB2-extrapolated depth-integral).
        # F_slow_eff = (3/2+ε)·F_slow^n − (1/2+ε)·F_slow^{n-1}, applied to the
        # barotropic forcing ONLY (du_dt_pert above keeps the current-time
        # baroclinic perturbation).  The ε = ab2_epsilon robustification is the
        # SAME χ-stabilised AB2 the outer baroclinic integrator uses (a_n/a_p at
        # ~L4093) and that Oceananigans applies to Gᵁ — bare (3/2,1/2) AB2 is the
        # marginally-unstable form, so robustifying matters on this barotropic
        # mode.  Cold start (prev=zeros) gives (3/2+ε)·F_slow ≈ 1.6× on step 1 —
        # the SAME first-step convention as the outer AB2 (not forward-Euler).
        # ``_F_slow_*_cur`` (current values) are stored as next-step prev below.
        # Default off ⇒ bit-identical (the branch is not traced).
        _F_slow_u_cur = F_slow_u
        _F_slow_v_cur = F_slow_v
        if getattr(_cfg_b.barotropic, "barotropic_slow_forcing_ab2", False):
            _fpu = state.F_slow_u_prev
            _fpv = state.F_slow_v_prev
            if _fpu is None or _fpv is None:
                # `None` is a static pytree-structure value (not traced), so this
                # raises cleanly at trace time with an actionable message instead
                # of a cryptic lax.scan "carry structure changed" error: the
                # storage block below ALWAYS writes a Field, so an unseeded
                # first step (prev=None) would flip None→Field between scan
                # iterations.  The driver MUST seed zeros Fields (see
                # build_silvestri_baroclinic_jet_setup).
                raise ValueError(
                    "barotropic_slow_forcing_ab2=True requires the state's "
                    "F_slow_u_prev/F_slow_v_prev to be seeded (zeros Field) "
                    "before stepping: the AB2 needs a stable pytree carry under "
                    "lax.scan (a None→Field transition breaks the scan). Seed "
                    "them as in build_silvestri_baroclinic_jet_setup.")
            _eps = _cfg_b.ab2_epsilon
            _an, _ap = 1.5 + _eps, 0.5 + _eps
            F_slow_u = (_an * F_slow_u - _ap * _fpu.data) * state.u_mask.data
            F_slow_v = (_an * F_slow_v - _ap * _fpv.data) * state.v_mask.data

        # Outer baroclinic momentum integrator (NEMO-mirror, #RK3).  The
        # cold-start amplifiers (pressure gradient + KE gradient + relative
        # vorticity flux) live in ``du_dt`` and are integrated explicitly
        # here; forward-Euler has no stability region for them, so the violent
        # geostrophic adjustment from rest amplifies.  SSP-RK3 (Shu-Osher)
        # mirrors NEMO's RK3 outer step.  T,S,eta + surface forcing are frozen
        # across the 3 stages (operator-split with the Matsuno Coriolis +
        # barotropic + tracer stages below); the barotropic slow forcing
        # F_slow_{u,v} is the stage-1 value already computed above.  Static
        # Python branch on the config string (no retrace / no jnp.where).
        # NOTE: the MOMENTUM stepping uses ``dt_mom`` (= dt / dt_mom_ratio) for
        # Veros-faithful asynchronous dt_mom≠dt_tracer stepping; the inner
        # ``self.tendencies(..., dt=dt)`` call keeps ``dt`` (= dt_tracer), which
        # is the tracer flux-limiter timestep.  dt_mom_ratio=1.0 (default) ⇒
        # dt_mom == dt ⇒ bit-identical for all existing configs.
        _nemo_ws_velocity_stages = None
        _nemo_ws_live_stage_geometry = None
        _nemo_ws_exposed_momentum_stage = None
        _nemo_ws_exposed_stage2_raw = None
        _nemo_ws_exposed_tracer_stage = None
        _nemo_ws_exposed_tracer_boundary = None
        _nemo_ws_exposed_tracer_transport = None
        _nemo_ws_exposed_stage_face_r3 = None
        _nemo_ws_exposed_stage1_wzv = None
        _nemo_ws_exposed_stage1_transport_operand = None
        _nemo_ws_exposed_momentum_operator = None
        _nemo_ws_exposed_stage1_rhs = None
        _nemo_ws_exposed_stage1_raw = None
        _nemo_ws_exposed_stage2_rhs = None
        _nemo_ws_exposed_stage3_rhs = None
        _nemo_ws_exposed_stage3_raw = None
        _nemo_ws_stage_tracers = None
        _nemo_ws_live_stage_states = None; _nemo_ws_live_stage_raw = None; _nemo_ws_live_stage_rhs = None; _nemo_ws_live_baro_geometry = None  # noqa: E501,E702
        _nemo_ws_live_slow_forcing_producer = None
        _nemo_ws_live_stage1_full_rhs = None; _nemo_ws_live_stage1_rhs_walk = None  # noqa: E702
        _nemo_ws_live_stage_qco = None
        _nemo_ws_live_tke_entry = None
        _nemo_ws_live_tke_statement_trace = None
        _nemo_ws_live_operator_operands = [
            _nemo_ws_stage1_operator_operands, None, None]
        _nemo_ws_tracer_content_rhs = None
        _nemo_ws_advection_content_rhs = None
        _nemo_ws_zdf_eta_kmm = None
        _nemo_ws_process_qco = None
        _nemo_ws_process_surface_rate = None
        _nemo_ws_process_qsr_rate = None
        _nemo_ws_qsr_association = None
        _nemo_ws_process_boundaries = None
        _nemo_ws_process_Taa = None; _nemo_ws_ldf_diagnostics = None
        _nemo_ws_vertical_solve_trace, _return_vertical_solve_trace = None, (_return_tracer_process_trace and self._nemo_ws_test_hooks.vertical_solve_trace)
        if getattr(_cfg_b, "momentum_time_integrator", "euler") == "rk3":
            u0 = state.u.data
            v0 = state.v.data
            # _geom_density (frozen EOS/pressure/geometry) computed once above
            # for the stage-1 tendency; reused here for every RK3 momentum
            # sub-stage (#25) so _bc_geometry_and_density is not recomputed.

            def _mom_pert(u_in, v_in):
                st = state._replace(
                    u=state.u.replace(data=u_in * u_mask_3d),
                    v=state.v.replace(data=v_in * v_mask_3d),
                )
                # momentum_only: T/S/eta are frozen across the RK3 momentum
                # sub-stages, so the tracer-diffusion tendency is recomputed
                # identically and discarded here (only du/dv are used). Skip it
                # -> bit-identical momentum, fewer halos/compute per sub-stage.
                # precomputed_geom_density: reuse the frozen EOS/pressure (#25).
                td = self.tendencies(st, surface_forcing, sponge=sponge, dt=dt,
                                     momentum_only=True,
                                     precomputed_geom_density=_geom_density,
                                     grid=_grid, vertex_mask=_vmask, z_coord=z_coord, config=config)
                _du = td.du_dt.data
                _dv = td.dv_dt.data
                _Fu = jnp.sum(_du * h_u_pre, axis=-1) / H_u_pre * state.u_mask.data
                _Fv = jnp.sum(_dv * h_v_pre, axis=-1) / H_v_pre * state.v_mask.data
                return _du - _Fu[..., jnp.newaxis], _dv - _Fv[..., jnp.newaxis]

            # Stage 1 perturbation = du_dt_pert/dv_dt_pert (computed above).
            u1 = u0 + dt_mom * du_dt_pert
            v1 = v0 + dt_mom * dv_dt_pert
            p1u, p1v = _mom_pert(u1, v1)
            u2 = 0.75 * u0 + 0.25 * (u1 + dt_mom * p1u)
            v2 = 0.75 * v0 + 0.25 * (v1 + dt_mom * p1v)
            p2u, p2v = _mom_pert(u2, v2)
            u_star = (1.0 / 3.0) * u0 + (2.0 / 3.0) * (u2 + dt_mom * p2u)
            v_star = (1.0 / 3.0) * v0 + (2.0 / 3.0) * (v2 + dt_mom * p2v)
        elif getattr(_cfg_b, "momentum_time_integrator",
                     "euler") == "rk3_ws":
            # NEMO WS-RK3 restarts each stage from u0 with the prior stage RHS
            # and dt/3, dt/2, dt (not Shu-Osher combinations). Stage 1 carries
            # the full stp2d RHS, stage 2 hpg+vor+adv, and stage 3 the full RHS
            # plus its only implicit ZDF solve. Its stability polynomial is
            # unchanged from the equivalent three-stage polynomial used here:
            # SSP-RK3's, so the explicit Coriolis bound still applies.
            u0 = state.u.data
            v0 = state.v.data

            def _mom_pert_ws(
                u_in, v_in, skip_ldf, transport_mean=None,
                stage_tracers_eta=None, extra_rhs=None,
                stage_face_thickness=None, stage_index=0,
                stage_zad_operands=None,
            ):
                st = state._replace(
                    u=state.u.replace(data=u_in * u_mask_3d),
                    v=state.v.replace(data=v_in * v_mask_3d),
                )
                stage_geom_density = _geom_density
                if stage_tracers_eta is not None:
                    stage_T, stage_S, stage_eta = stage_tracers_eta
                    st = st._replace(
                        T=state.T.replace(data=stage_T),
                        S=state.S.replace(data=stage_S),
                        eta=state.eta.replace(data=stage_eta),
                    )
                    stage_geom_density = compute_frozen_geom_density(
                        st, _grid, _zc, _cfg_b)
                transport_velocity = None
                if transport_mean is not None:
                    transport_u_mean, transport_v_mean = transport_mean
                    current_u_mean = (
                        jnp.sum(u_in * h_u_pre, axis=-1) / H_u_pre
                        * state.u_mask.data)
                    current_v_mean = (
                        jnp.sum(v_in * h_v_pre, axis=-1) / H_v_pre
                        * state.v_mask.data)
                    # Validated at construction (see
                    # ``_STAGE1_TRANSPORT_OPERAND_ARMS``), so only the legal
                    # strings reach here.
                    _tr_arm = (self._nemo_ws_test_hooks
                               .momentum_transport_stage1_operand)
                    if _tr_arm and stage_index == 1:
                        if _tr_arm == "prognostic_mean":
                            # stprk3_stg.f90:270 subtracts uu_b(:,:,Kmm),
                            # the external mode's own prognostic, NOT a
                            # depth mean re-reduced from the 3-D velocity.
                            if state.uu_b is None or state.vv_b is None:
                                raise ValueError(
                                    "momentum_transport_stage1_operand="
                                    "'prognostic_mean' needs the NEMO "
                                    "prognostic uu_b/vv_b pair; this state "
                                    "has none, and falling through would "
                                    "report the production path under the "
                                    "arm's name")
                            current_u_mean = (state.uu_b.data
                                              * state.u_mask.data)
                            current_v_mean = (state.vv_b.data
                                              * state.v_mask.data)
                        else:
                            # The pre-round-206 behaviour, kept as the
                            # EXACT one-variable control of what round 206
                            # landed: the target built on the sum of the
                            # MIN-RULE face thicknesses instead of NEMO's
                            # ``hu_0*(1+r3u(Kmm))`` (stprk3_stg.f90:270).
                            # Taken as the whole array rather than undone by
                            # a second multiply, so the arm is a BITWISE
                            # revert of the landed statement and goes
                            # exactly inert if that statement is reverted --
                            # which is what makes the test that asserts it
                            # moves the step a pin and not a smoke test.
                            transport_u_mean, transport_v_mean = (
                                _transport_target_legacy())
                    # stprk3_stg.F90:273-274: the SAME barotropic correction
                    # enters the advective transport masked by the 3-D
                    # umask/vmask -- ``zFu = e2u*e3u(Kmm)*( uu(Kmm) +
                    # zub*umask(ji,jj,jk) )`` -- and this array is what
                    # dyn_adv_up3 consumes as its transport.
                    transport_velocity = (
                        (u_in + (transport_u_mean - current_u_mean)[..., None])
                        * _ws_stage_u_mask,
                        (v_in + (transport_v_mean - current_v_mean)[..., None])
                        * _ws_stage_v_mask,
                    )
                _operator_stage = (
                    self._nemo_ws_test_hooks.expose_momentum_operator_stage)
                _expose_operator = (
                    stage_index == _operator_stage
                    and bool(self._nemo_ws_test_hooks.expose_momentum_operator))
                _return_components = (
                    _expose_operator or _return_live_stage_operands)
                _zad_observers = (None,
                    self._nemo_ws_test_hooks.stage2_zad_operand_observer,
                    self._nemo_ws_test_hooks.stage3_zad_operand_observer)
                _stage_ldf_thickness = None
                if (not skip_ldf and _ws_uses_nemo_ldf_e3
                        and stage_face_thickness is not None):
                    from legoesm.ocean.vertical import (
                        nemo_ldf_reference_e3f,
                        nemo_qco_live_vorticity_e3f_cgrid,
                    )
                    # Same consumer-local reference as the stage-1 call above
                    # (dynldf_lev.f90:123); the stage only moves r3f.
                    _stage_ldf_thickness = (
                        _geom_density[1], _ws_ldf_face_thickness_kbb[0],
                        _ws_ldf_face_thickness_kbb[1],
                        nemo_qco_live_vorticity_e3f_cgrid(
                            st.eta.data, _zc, st.eta.data.dtype, grid=_grid,
                            e3t_0=_ws_h_ref, tmask=_ws_t_live_mask,
                            reference_e3f=nemo_ldf_reference_e3f(_zc)),
                        stage_face_thickness[0], stage_face_thickness[1])
                td_result = self.tendencies(
                                     st, surface_forcing, sponge=sponge, dt=dt,
                                     momentum_only=True,
                                     precomputed_geom_density=stage_geom_density,
                                     grid=_grid, vertex_mask=_vmask,
                                     skip_lateral_viscosity=skip_ldf,
                                     # dynldf_lev_rot_scheme.h90:24-25 and
                                     # :28-29 read pu_in(ji,jj,jk,Kbb) /
                                     # pv_in(ji,jj,jk,Kbb): the lateral
                                     # Laplacian is evaluated on the BEFORE
                                     # velocity, not on the stage velocity the
                                     # rest of the RHS uses.  Kbb is the
                                     # step-entry level at every WS-RK3 stage
                                     # (stprk3_stg.F90:118,174,218), and
                                     # dynldf.F90:70 hands the operator the
                                     # whole array with that index.  Stage 1
                                     # passes u_in = u0 already, so only the
                                     # stage-3 call moves.  The tracer pair is
                                     # the stage's own, because this call is
                                     # momentum_only and tra_ldf is a separate
                                     # NEMO routine (stprk3_stg.F90:586).
                                     ldf_state=(
                                         st.T.data, st.S.data,
                                         u0 * u_mask_3d, v0 * v_mask_3d),
                                     z_coord=z_coord, config=config,
                                     momentum_flux_transport_velocity=(
                                         transport_velocity),
                                     # dynadv_up3.F90:166-170: the stage RHS
                                     # selects the UP3 branch by the scheme's
                                     # own reference rule whether or not a
                                     # separate transport is supplied (so the
                                     # gate's transport-reconcile arm stays
                                     # one-variable); the private hook
                                     # restores the legacy transport sign.
                                     up3_upwind_selector=_up3_selector_override,
                                     # stprk3_stg.F90:273 / dynadv_up3.F90:
                                     # 205-207: the stage's e3u/e3v(Kmm)
                                     # from _nemo_ws_qco_stage_faces keyed on
                                     # the stage ssh (None under the legacy
                                     # thickness hook).
                                     momentum_flux_face_thickness=(
                                         None if (
                                             self._nemo_ws_test_hooks
                                             .legacy_vector_ene_min_face_thickness
                                         ) else stage_face_thickness),
                                     ldf_thickness_operands=(
                                         _stage_ldf_thickness),
                                     ene_generic_f_vtx=(
                                         self._nemo_ws_test_hooks
                                         .legacy_ene_vertex_coriolis),
                                     legacy_hpg_algebraic=(
                                         self._nemo_ws_test_hooks
                                         .legacy_hpg_algebraic_association),
                                     nemo_operator_association=(
                                         self._nemo_ws_test_hooks
                                         .nemo_stage_rhs_accumulation_order_arm is True),
                                     nemo_stage_zad_operands=stage_zad_operands,
                                     nemo_stage_zad_operand_observer=(
                                         _zad_observers[stage_index - 1]),
                                     return_nemo_operator_components=(
                                         _return_components))
                if _return_components:
                    td, _, _operator_components = td_result
                    if (_return_live_stage_operands
                            and stage_index in (2, 3)):
                        _nemo_ws_live_operator_operands[stage_index - 1] = (
                            _operator_components)
                else:
                    td = td_result
                if _expose_operator:
                    _operator_name = (
                        self._nemo_ws_test_hooks.expose_momentum_operator)
                    if _operator_name not in (
                            "hpg", "vorticity", "advection", "keg", "zad", "after_hpg", "after_vor", "after_keg", "after_zad"):
                        raise ValueError(
                            "expose_momentum_operator must name a component "
                            "or after_* accumulator boundary")
                    _op_u = _operator_components[f"{'after_adv' if _operator_name == 'after_zad' else _operator_name}_u"].data
                    _op_v = _operator_components[f"{'after_adv' if _operator_name == 'after_zad' else _operator_name}_v"].data
                    if (_operator_name in ("keg", "zad")
                            and extra_rhs is not None):
                        # The two halves are the VECTOR form's routines.  A
                        # flux-form card adds its own vertical term to the
                        # same dyn_adv bucket through ``extra_rhs``, and that
                        # term belongs to neither half, so publishing a half
                        # there would silently omit it.  Refuse instead.
                        raise ValueError(
                            "expose_momentum_operator='keg'/'zad' is the "
                            "vector-invariant split; this stage carries a "
                            "separate flux-form vertical term")
                    if _operator_name == "advection" and extra_rhs is not None:
                        # dynadv_up3's stage-3-style vertical term is evaluated
                        # by the shared literal helper and added to the same
                        # dyn_adv source bucket as the kernel's KEG+ZAD terms.
                        _op_u = _op_u + extra_rhs[0]
                        _op_v = _op_v + extra_rhs[1]
                    nonlocal _nemo_ws_exposed_momentum_operator
                    _nemo_ws_exposed_momentum_operator = (_op_u, _op_v)
                _du = td.du_dt.data
                _dv = td.dv_dt.data
                if extra_rhs is not None:
                    _du = _du + extra_rhs[0]
                    _dv = _dv + extra_rhs[1]
                if not self._nemo_ws_test_hooks.legacy_preproject_stage_rhs:
                    # NEMO first forms Kaa from the FULL Krhs (:396-419), then
                    # removes its reference-thickness column mean and replaces
                    # it with uu_b/vv_b (:433-446).  Pre-projecting Krhs is
                    # algebraically equivalent, but changes rounding at the
                    # stage-2 boundary on GYRE's live ENE depth-mean gauge.
                    return _du, _dv
                _Fu = jnp.sum(_du * h_u_pre, axis=-1) / H_u_pre * state.u_mask.data
                _Fv = jnp.sum(_dv * h_v_pre, axis=-1) / H_v_pre * state.v_mask.data
                return _du - _Fu[..., jnp.newaxis], _dv - _Fv[..., jnp.newaxis]

            # The barotropic solve is seeded with the BEFORE velocity, not with
            # a momentum ladder.  NEMO's stp_2D evaluates the Kbb RHS ONCE
            # (eos/dyn_hpg/dyn_ldf/dyn_vor/wzv/dyn_adv at stp2d.F90:126-171),
            # depth-means it into Ue_rhs/Ve_rhs (:177-186) and then calls
            #   CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ... )
            # at stp2d.F90:280-281 -- Kmm == Kbb, i.e. uu at the BEFORE level,
            # with no momentum stage in between.  The depth mean of that single
            # RHS is what F_slow_u/F_slow_v already carry into the solver here.
            # The WS stage recurrence belongs to stprk3_stg (stprk3_stg.F90:
            # 344-374) and now runs exactly once, after the barotropic solve,
            # where it can use the actual external-mode velocity.  It used to be
            # written a SECOND time at this point purely to produce this seed:
            # two extra `tendencies()` evaluations per step, and a fix applied
            # to one copy silently left the other stale.
            u_star = u0
            v_star = v0
        else:
            u_star = state.u.data + dt_mom * du_dt_pert
            v_star = state.v.data + dt_mom * dv_dt_pert

        # 4. Forward-backward Coriolis on perturbation velocity
        #
        # Forward Euler Coriolis amplifies inertial oscillations by
        # sqrt(1 + (f*dt)^2) per step, causing blowup at high latitudes.
        # Forward-backward (Matsuno) stepping is unconditionally neutral:
        #   u' += dt * f * v'_at_u          (forward: old v')
        #   v' -= dt * f * u'_new_at_v      (backward: new u')
        # This matches the barotropic solver's Coriolis treatment.
        #
        # SKIPPED under coriolis_scheme="explicit_ab2" (Veros-faithful): there
        # the plain f×u entered tend.du_dt/dv_dt inside ``tendencies`` above, so
        # its perturbation is already carried in du_dt_pert → u_star (and AB2'd
        # by _ab2_step), and its depth-mean is in F_slow → the barotropic solve.
        # Applying the Matsuno rotation here too would DOUBLE-count Coriolis.
        # Static Python branch on the config string ⇒ default is bit-identical.
        if getattr(_cfg_b, "coriolis_scheme", "matsuno_split") != "explicit_ab2":
            u_star, v_star = _forward_backward_coriolis_3d(
                u_star, v_star, dt_mom, _grid, _zc, _cfg_b,
                state.u_mask.data, state.v_mask.data, state.land_mask.data,
                state.eta.data, state.H_bathy.data,
            )

        # Enforce periodic wrap column: u[:,n_lon] must equal u[:,0].
        u_star = u_star.at[:, -1].set(u_star[:, 0])

        state_mid = state._replace(
            u=state.u.replace(data=u_star * u_mask_3d),
            v=state.v.replace(data=v_star * v_mask_3d),
            T=state.T.replace(data=T_new * mask_3d),
            S=state.S.replace(data=S_new * mask_3d),
        )

        # 5. Save pre-barotropic layer thickness
        h_k_old = compute_layer_thickness(
            state_mid.eta.data, state_mid.H_bathy.data, _zc,
            min_water_column_m=_cfg_b.min_water_column_m,
        )

        # 6. Barotropic step.  Two paths:
        #    - explicit_substep: split-explicit forward-backward substepping
        #      with cosine/box time filter.
        #    - implicit_cn: single-step Crank-Nicolson free surface (PCG).
        #      Eliminates the chequerboard mode by construction; no
        #      substepping or time filter needed.

        # Freshwater mass flux for barotropic continuity equation
        F_slow_eta = None
        _F_fw_rate_now = None
        if freshwater is not None and _cfg_b.freshwater_closure != "none":
            F_slow_eta = freshwater_eta_tendency(
                freshwater, _cfg_b.rho_0,
            ) * state.land_mask.data
            if (_cfg_b.freshwater_closure == "real_freshwater"
                    and bool(getattr(_cfg_b, "normalize_freshwater",
                                     False))):
                # codex RED: lat-lon's ONLY freshwater normalizer lives inside
                # the virtual-salt block, which `real_freshwater` skips -- so
                # without this, normalize_freshwater=True silently became a
                # no-op in the new mode.  Normalize the eta/VOLUME forcing
                # instead, which is what the flag must mean once there is no
                # virtual-salt channel to normalize.  Scoped to real mode ONLY,
                # so virtual-mode runs stay bit-identical.
                #
                # REUSE the shared normalizer rather than a hand-rolled mean
                # (codex RED round 2): it already carries the lat-band SPMD
                # `psum` and the MPI owned-mask reduction.  My first version
                # took a RANK-LOCAL mean off the GLOBAL grid.area_T, which both
                # shape-mismatches the injected band grid under lat SPMD and
                # would subtract a per-band mean.
                from legoesm.ocean.freshwater import normalize_freshwater_net
                _g_eta = _grid if _grid is not None else self.grid
                # RESTORING IS EXCLUDED FROM THE NORMALIZATION (codex round 2
                # RED).  The flag exists to remove the CORE-II P-E+R imbalance,
                # a forcing-dataset artifact.  SSS restoring is not part of
                # that imbalance, and NEMO never normalizes its `erp`
                # (sbcssr.F90 adds it straight to `emp`; the global correction
                # in NEMO applies to the forcing fields, not the damping term).
                # Normalizing the full net would subtract the restoring's own
                # global mean from EVERY cell -- a spurious uniform water flux
                # plus a globally weakened restoring, both silent.
                #
                # So: normalize the PHYSICAL net, then add the restoring rate
                # back untouched.  When `restoring` is absent this is exactly
                # the previous expression.
                _fw_rest = getattr(freshwater, "restoring", None)
                if _fw_rest is None:
                    F_slow_eta = normalize_freshwater_net(
                        F_slow_eta, _g_eta.area_T, state.land_mask.data)
                else:
                    _F_rest = (jnp.asarray(_fw_rest) / _cfg_b.rho_0
                               ) * state.land_mask.data
                    F_slow_eta = normalize_freshwater_net(
                        F_slow_eta - _F_rest, _g_eta.area_T,
                        state.land_mask.data) + _F_rest
            # barotropic_forcing_centred (#1226 item 3; NEMO ln_bt_fw=.FALSE.,
            # dynspg_ts.F90:415-421 ssh_frc = ((emp+emp_b) -
            # (rnf+rnf_b))/(2*rho0)): centre ONLY this eta/barotropic channel — NEMO's
            # emp_b/rnf_b enter ssh_frc, never the separate tra_sbc tracer
            # deposit (lego's virtual_salt_flux, called later from the SAME
            # ``freshwater`` arg, stays at NOW/uncentred). Surgical (not a
            # ``freshwater`` rebind) so the tracer channel is untouched.
            # ``freshwater_eta_prev`` is the ALREADY-REDUCED previous
            # F_slow_eta rate (same reduction, so the average is linear-exact
            # vs averaging the raw FreshwaterForcing first). None seeding:
            # same nit000 rule as the wind term above.
            # The tracer dilution channel (block 8, real_freshwater) takes
            # this NOW rate: NEMO centres only ssh_frc, tra_sbc stays at NOW.
            _F_fw_rate_now = F_slow_eta
            if (getattr(_cfg_b, "barotropic_forcing_centred", False)
                    and getattr(state, "freshwater_eta_prev", None)
                    is not None):
                F_slow_eta = 0.5 * (state.freshwater_eta_prev + F_slow_eta)

        # ab2_scope="advective": the dissipative momentum tendencies (lateral
        # friction + bottom drag) are withheld from du_dt for weight-1.0
        # placement, but their DEPTH-MEAN must still force the barotropic
        # solve — Veros's streamfunction forcing is the depth-integral of
        # (du[tau] + du_mix) (solve_stream.py:80) and friction.py routes both
        # bottom drag and lateral friction through du_mix. Without this the
        # drag never reaches the persistent barotropic balance (the rigid lid
        # replaces the incoming depth-mean every step) and the transport runs
        # away (measured 3865 Sv vs 248 — adversarial review check 3). Added
        # HERE (after du_dt_pert was formed from the diss-free F_slow, so the
        # 3-D perturbation stays uncontaminated); the weight-1.0 3-D
        # application then adds only the BAROCLINIC deviation.
        if tend.du_diss is not None:
            F_slow_u = (F_slow_u + jnp.sum(
                tend.du_diss.data * h_u_pre, axis=-1) / H_u_pre
                ) * state.u_mask.data
            F_slow_v = (F_slow_v + jnp.sum(
                tend.dv_diss.data * h_v_pre, axis=-1) / H_v_pre
                ) * state.v_mask.data

        if _cfg_b.barotropic.barotropic_solver == "rigid_lid":
            # Rigid lid: no free surface — solve the barotropic streamfunction
            # from the (Coriolis-augmented) slow forcing.  eta is left unchanged.
            # Freshwater (F_slow_eta) cannot change a rigid lid's volume; it
            # enters elsewhere as a virtual salt flux.
            from legoesm.ocean.dynamics.rigid_lid_latlon_cgrid import (
                barotropic_rigid_lid_latlon_cgrid,
            )
            rl_data = self._ensure_rigid_lid_data(state_mid)
            # Under coriolis_scheme="explicit_ab2" the planetary Coriolis is
            # already inside F_slow_u/v (its depth-mean came through du_dt); the
            # solver must NOT add its own f×u_bt again (no double count).
            _add_bt_cor = (
                getattr(_cfg_b, "coriolis_scheme", "matsuno_split")
                != "explicit_ab2")
            state_new, (Hu_avg, Hv_avg) = barotropic_rigid_lid_latlon_cgrid(
                state_mid, dt_mom, _grid, _zc, _cfg_b, rl_data,
                F_slow_u=F_slow_u, F_slow_v=F_slow_v,
                add_barotropic_coriolis=_add_bt_cor,
            )
        elif _cfg_b.barotropic.barotropic_solver == "implicit_cn":
            state_new, (Hu_avg, Hv_avg) = barotropic_implicit_latlon_cgrid(
                state_mid, dt_mom,
                _grid, _zc, _cfg_b,
                F_slow_eta=F_slow_eta,
                F_slow_u=F_slow_u,
                F_slow_v=F_slow_v,
            )
        else:
            # ``_barotropic_substep_scale`` (leap-frog): the config's
            # ``n_barotropic_substeps`` is calibrated for a dt-length baroclinic
            # window (the forward-Euler path); the leap-frog integrates the
            # barotropic mode over rDt=2dt, so WITHOUT rescaling the substep length
            # DOUBLES (30 substeps over 2dt = 180 s vs the CFL-safe 90 s) and the
            # split-explicit free surface goes unstable in ~4 steps. Scaling the
            # count by rDt/dt keeps the substep length (and the barotropic CFL)
            # identical to the FE path. Default 1 ⇒ every other caller unchanged.
            _nbaro = (_cfg_b.barotropic.n_barotropic_substeps
                      * _barotropic_substep_scale)
            dt_s = dt_mom / _nbaro
            # Under coriolis_scheme="explicit_ab2" the planetary Coriolis already
            # reaches the barotropic mode via F_slow (its depth-mean came through
            # du_dt), so the substep must NOT add its own f×U_bt — this is the
            # Oceananigans split-explicit convention and removes the C-grid
            # 4-point Coriolis rotational null mode (the 2Δx barotropic mode that
            # otherwise blows the eddy-resolving jet).
            _add_bt_cor = (
                getattr(_cfg_b, "coriolis_scheme", "matsuno_split")
                != "explicit_ab2")
            _een_pre_shared = None
            if (not _add_bt_cor) and getattr(
                    _cfg_b, "barotropic_coriolis_split",
                    "frozen") == "live":
                # NEMO dynspg_ts structure (dynspg_ts.F90:296-300 + :689):
                # remove the PRE-step 2D barotropic Coriolis from the frozen
                # forcing (leaving baroclinic-only F_slow, NEMO's zu_frc), then
                # integrate a LIVE f x U every substep so the barotropic mode
                # holds geostrophic balance with the evolving eta inside the
                # window (the frozen form lags Coriolis by dt and cripples the
                # gyre's U_bar response to grad-eta). The subtraction uses the
                # SAME stencil the substep loop applies live so it cancels at the
                # pre-step state: EEN (barotropic_coriolis="een", the DINO/NEMO
                # ln_dynvor_een form) or the legacy 4-pt V-to-u average
                # (interp_u_to_vface_4pt). The 4-pt cancellation is EXACT on a
                # beta-plane with flat full-cell bathymetry (average and
                # depth-mean commute), else an O(dx^2)/topographic residual
                # remains; the EEN path cancels exactly (same helper both sides).
                _bt_cor_split = getattr(
                    _cfg_b.barotropic, "barotropic_coriolis", "avg")
                if _bt_cor_split in ("ene", "ene_metric", "een", "een_metric"):
                    # NEMO ln_dynvor_een DINO (nemo_dino_kamm_mlf): the _total
                    # planetary term rides the vertex-f EEN transport-form flux,
                    # so the pre-step subtraction must use the SAME EEN stencil
                    # the substep loop applies live (dyn_cor_2D). Built from the
                    # Nnn thickness (h_k_pre, = state_mid.eta since the barotropic
                    # solve has not yet updated eta) and the POST-slow-tendency
                    # barotropic velocity (state_mid.u/v = NOW/Kmm) — matching
                    # NEMO, which subtracts the Kmm barotropic Coriolis from zu_frc
                    # (dynspg_ts.F90:359) and re-applies dyn_cor_2D LIVE (:689) on
                    # the evolving transport.  NB under the leap-frog (residual #1)
                    # the substep loop is SEEDED from the BEFORE level (Nbb), so at
                    # substep 0 the live term acts on the Nbb transport while this
                    # subtraction removed the Nnn/Kmm Coriolis — they do NOT cancel
                    # bit-exactly; the O(f·(U_Nnn−U_Nbb)) residual IS the leap-frog
                    # evolution (this is NEMO's design: subtract at Kmm, seed at
                    # Kbb), not a double-count.  Under forward_euler the seed IS Nnn
                    # so it cancels exactly.  cor_v already carries the -f*U sign, so
                    # both are SUBTRACTED (unlike the 4-pt branch's explicit +f*U).
                    # COEFFICIENT time level (#1226 item 4): this subtraction's EEN
                    # coefficients are built from h_k_pre (Kmm/NOW).  NEMO uses the
                    # SAME dyn_cor_2D_init(Kmm) coefficients for BOTH the :359
                    # subtraction and the :689 live substep application; lego's
                    # in-substep coefficients match only under
                    # barotropic_een_seed="nemo_kmm" (the legacy "window_start"
                    # builds them from the Nbb seed thickness under the MLF
                    # before-level seed — an operator mismatch NEMO doesn't have).
                    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
                        barotropic_coriolis_een_pre_step,
                    )
                    _min_wc = jnp.asarray(
                        _cfg_b.min_water_column_m, dtype=F_slow_u.dtype)
                    _een_eval = getattr(
                        _cfg_b.barotropic,
                        "barotropic_een_coefficient_evaluation", "generic")
                    _bt_pv_scheme = (
                        "ene" if _bt_cor_split in ("ene", "ene_metric")
                        else "een")
                    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
                        nemo_carried_barotropic_depth_mean,
                    )
                    # The SAME selector the window seed uses: a card that
                    # selects NEMO's carried external mode and carries no
                    # pair (or half a pair) RAISES rather than falling back
                    # to a reduction, which is the silent fallback this
                    # round is removing in the first place.
                    _nemo_carried_baro = nemo_carried_barotropic_depth_mean(
                        state, F_slow_u.dtype, _cfg_b)
                    _slow_incoming_override = (
                        self._nemo_ws_test_hooks
                        .slow_forcing_incoming_override)
                    if _slow_incoming_override is not None:
                        F_slow_u, F_slow_v = _slow_incoming_override
                    _slow_incoming_u = F_slow_u
                    _slow_incoming_v = F_slow_v
                    (_cor_u_sub, _cor_v_sub,
                     _een_pre_built) = barotropic_coriolis_een_pre_step(
                        state_mid.u.data, state_mid.v.data, h_k_pre, _grid,
                        state.land_mask.data, state.u_mask.data,
                        state.v_mask.data, _min_wc, F_slow_u.dtype,
                        metric_complete=_bt_cor_split.endswith("_metric"),
                        # Same EEN q-boundary / e3f rules as the 3-D EEN and
                        # as the substep loop's own _build_een_barotropic_inputs
                        # — otherwise this subtraction uses a DIFFERENT operator
                        # than the live term it is meant to cancel.
                        een_q_boundary=getattr(
                            _cfg_b, "een_q_boundary", "neumann_fill"),
                        een_e3f_scheme=getattr(
                            _cfg_b, "een_e3f_scheme", "min"),
                        # ...and the SAME dz_ref, or the fully-dry-vertex e3f
                        # differs from the live substep term this cancels.
                        dz_ref=getattr(_zc, "dz_ref", None),
                        coefficient_evaluation=_een_eval,
                        eta=state.eta.data,
                        z_coord=_zc,
                        return_pre=True,
                        scheme=_bt_pv_scheme,
                        # dyn_cor_2D, CALLED at dynspg_ts.f90:289 and
                        # subtracted at :292, is handed puu_b(:,:,Kmm) --
                        # the carried external mode, the SAME array the
                        # substep loop seeds from (dynspg_ts.F90:484-500).
                        # stprk3.f90:189 calls stp_2D(kstp, Nbb, Nbb, ...),
                        # so Kmm IS Nbb here.  Not a fresh reduction of the
                        # 3-D velocity.  Cards that do not run the carried
                        # external mode keep the reduction.
                        entry_barotropic_velocity=_nemo_carried_baro)
                    if _een_eval == "nemo_literal":
                        _een_pre_shared = _een_pre_built
                    F_slow_u = (F_slow_u - _cor_u_sub) * state.u_mask.data
                    F_slow_v = (F_slow_v - _cor_v_sub) * state.v_mask.data
                    _slow_final_override = self._nemo_ws_test_hooks.barotropic_slow_forcing_override
                    if callable(_slow_final_override):
                        jax.debug.callback(
                            _slow_final_override, _slow_incoming_u, _slow_incoming_v,
                            _cor_u_sub, _cor_v_sub, state.u_mask.data, state.v_mask.data,
                            F_slow_u, F_slow_v, ordered=True)
                    elif _slow_final_override is not None:
                        F_slow_u, F_slow_v = _slow_final_override
                    _nemo_ws_live_slow_forcing_producer = {
                        "rhs_u": du_dt, "rhs_v": dv_dt,
                        "thickness_u": h_u_pre, "thickness_v": h_v_pre,
                        "depth_u": H_u_pre, "depth_v": H_v_pre,
                        "post_wind_u": _F_slow_wind_u,
                        "post_wind_v": _F_slow_wind_v,
                        "post_drag_u": _F_slow_drag_u,
                        "post_drag_v": _F_slow_drag_v,
                        "wind_tau_u": _wind_tau_i_u,
                        "wind_tau_v": _wind_tau_j_v,
                        "wind_r1_rho0": _wind_r1_rho0,
                        "incoming_u": _slow_incoming_u,
                        "incoming_v": _slow_incoming_v,
                        "coriolis_u": _cor_u_sub,
                        "coriolis_v": _cor_v_sub,
                        "final_u": F_slow_u,
                        "final_v": F_slow_v,
                    }
                    _add_bt_cor = True
                else:
                    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
                        interp_u_to_vface_4pt,
                    )
                    from legoesm.ocean.dynamics.barotropic_common import (
                        coriolis_at_faces,
                    )
                    _f_u2, _f_v2 = coriolis_at_faces(_grid, F_slow_u.dtype)
                    _u_pre3, _v_pre3 = state.u.data, state.v.data
                    _U_pre = (jnp.sum(_u_pre3 * h_u_pre, axis=-1)
                              / jnp.maximum(H_u_pre, 1e-10))
                    _V_pre = (jnp.sum(_v_pre3 * h_v_pre, axis=-1)
                              / jnp.maximum(H_v_pre, 1e-10))
                    _V_w = jnp.roll(_V_pre, 1, axis=1)
                    _V_at_u = 0.25 * (_V_pre[:-1] + _V_pre[1:]
                                      + _V_w[:-1] + _V_w[1:])
                    _V_at_u = jnp.concatenate([_V_at_u, _V_at_u[:, 0:1]], axis=1)
                    _U_at_v = interp_u_to_vface_4pt(_U_pre, _grid)
                    F_slow_u = (F_slow_u - _f_u2 * _V_at_u) * state.u_mask.data
                    F_slow_v = (F_slow_v + _f_v2 * _U_at_v) * state.v_mask.data
                    _add_bt_cor = True
            if _cfg_b.barotropic.barotropic_wide_halo:
                # Opt-in wide-halo subcycle: one fused wide exchange per
                # chunk of substeps instead of ~4 pads/substep (scaling-
                # audit item 3).  Static config gate — serial results are
                # value-identical; parity gated in
                # tests/ocean/unit/test_barotropic_wide_halo.py.
                from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
                    barotropic_substeps_wide_halo_latlon_cgrid,
                )
                if _barotropic_before_state is not None:
                    raise NotImplementedError(
                        "the MLF before-level barotropic seed is not wired into "
                        "the wide-halo barotropic path (the before eta/u/v would "
                        "need the extended-band widening); use the standard "
                        "split-explicit path (barotropic_wide_halo=False) with "
                        "outer_integrator in ('leapfrog', 'nemo_mlf').")
                _baro_fn = barotropic_substeps_wide_halo_latlon_cgrid
                _baro_seed = {}
            else:
                _baro_fn = barotropic_substeps_latlon_cgrid
                # MLF leap-frog: seed the barotropic integration from the
                # BEFORE level (Nbb) so the fast mode leap-frogs n-1 → n+1
                # (NEMO dynspg_ts.F90:494-503). Default (None) seeds from NOW.
                if _barotropic_before_state is not None:
                    _eta_bef, _u_bef, _v_bef = _barotropic_before_state
                    _baro_seed = dict(
                        eta_init=_eta_bef, u_init=_u_bef, v_init=_v_bef)
                else:
                    _baro_seed = {}
            # The MLF scale rescales the substep COUNT but must NOT widen the
            # NEMO boxcar averaging window: the half-width stays the unscaled
            # nn_e (= _nbaro / scale).  Only the standard path builds the boxcar
            # weights; the wide-halo twin refuses the EEN/boxcar MLF config.
            if _baro_fn is barotropic_substeps_latlon_cgrid:
                # TIME LEVEL of the barotropic bottom-drag RATE (#1455).  NEMO
                # builds rCdU_bot in zdf_phy from uu(:,:,:,Kmm) (zdfdrg.F90:
                # 174-181) at stpmlf.F90:190 -- BEFORE dyn_adv/vor/ldf/hpg/spg
                # -- and dyn_drg_init (dynspg_ts.F90:1616) freezes that same
                # array over the substep window.  The solver is handed
                # ``state_mid``, whose velocity is the POST-momentum u* (built
                # at the ``state_mid = state._replace(...)`` above), so the NOW
                # velocity has to travel separately.  These are the SAME arrays
                # the dyn_drg_init pu_RHSi residual above already reads.  NEMO
                # has ONE rCdU_bot; this model builds it in three places, and
                # all three are now on the now level -- the third, the implicit
                # vertical-mixing matrix, was fixed by the sibling commit that
                # threads the same u_now/v_now into
                # ``_apply_implicit_vertical_mixing`` (#1455).
                _baro_seed = dict(
                    _baro_seed, substep_scale=_barotropic_substep_scale,
                    u_now=state.u.data, v_now=state.v.data)
                if _een_pre_shared is not None:
                    _baro_seed = dict(
                        _baro_seed, een_pre_override=_een_pre_shared)
                if (
                    getattr(_cfg_b, "momentum_time_integrator", "euler")
                    == "rk3_ws"
                    and not self._nemo_ws_test_hooks.primary_transport_average
                ):
                    _baro_seed = dict(
                        _baro_seed,
                        _nemo_primary_transport_average_test_override=False)
                if _return_barotropic_substeps:
                    _baro_seed = dict(
                        _baro_seed, _nemo_substep_trace_test_hook=True)
                    _flux_update_override = (
                        self._nemo_ws_test_hooks
                        .barotropic_flux_form_update_override)
                    if _flux_update_override is not None:
                        _baro_seed = dict(
                            _baro_seed,
                            _nemo_flux_form_update_test_override=(
                                _flux_update_override))
                _cor_sub_override = (
                    self._nemo_ws_test_hooks
                    .barotropic_substep_coriolis_override)
                if _cor_sub_override is not None:
                    _baro_seed = dict(
                        _baro_seed,
                        _nemo_substep_coriolis_test_override=(
                            _cor_sub_override))
                _pgf_sub_override = (
                    self._nemo_ws_test_hooks.barotropic_substep_pgf_override)
                if _pgf_sub_override is not None:
                    _baro_seed = dict(
                        _baro_seed,
                        _nemo_substep_pgf_test_override=_pgf_sub_override)
                if self._nemo_ws_test_hooks.legacy_seed_min_rule_faces:
                    _baro_seed = dict(
                        _baro_seed,
                        _nemo_legacy_seed_faces_test_override=True)
                if (
                    self._nemo_ws_test_hooks
                    .barotropic_drag_rate_override is not None
                ):
                    _baro_seed = dict(
                        _baro_seed,
                        _nemo_drag_rate_test_override=(
                            self._nemo_ws_test_hooks
                            .barotropic_drag_rate_override))
                if (self._nemo_ws_test_hooks
                        .barotropic_raw_history_override is not None):
                    _baro_seed = dict(
                        _baro_seed,
                        _nemo_raw_history_test_override=(
                            self._nemo_ws_test_hooks
                            .barotropic_raw_history_override))
                if (
                    self._nemo_ws_test_hooks
                    .legacy_barotropic_continuity_association
                ):
                    _baro_seed = dict(
                        _baro_seed,
                        _nemo_continuity_update_test_override=False)
            elif _return_barotropic_substeps:
                raise NotImplementedError(
                    "GYRE substep trace requires the standard-halo barotropic path")
            elif self._nemo_ws_test_hooks.legacy_seed_min_rule_faces:
                raise ValueError(
                    "legacy_seed_min_rule_faces is a one-variable control of "
                    "the standard-halo explicit substep solver's loop-entry "
                    "seed; this configuration routes the external solve "
                    "elsewhere, so the control would silently not land.")
            _baro_result = _baro_fn(
                state_mid, dt_s, _nbaro,
                _grid, _zc, _cfg_b,
                F_slow_eta=F_slow_eta,
                F_slow_u=F_slow_u,
                F_slow_v=F_slow_v,
                add_barotropic_coriolis=_add_bt_cor,
                t_seconds=t_seconds,  # traced model time for the equilibrium tide
                **_baro_seed,
            )
            if _return_barotropic_substeps:
                state_new, (Hu_avg, Hv_avg), _substep_trace = _baro_result
                return _NEMOWSBarotropicTrace(
                    state_new, _substep_trace,
                    (F_slow_eta, F_slow_u, F_slow_v),
                    {
                        "du_dt": du_dt,
                        "dv_dt": dv_dt,
                        "h_u": h_u_pre,
                        "h_v": h_v_pre,
                        "H_u": H_u_pre,
                        "H_v": H_v_pre,
                        "depth_u": _F_slow_depth_u,
                        "depth_v": _F_slow_depth_v,
                        "post_wind_u": _F_slow_wind_u,
                        "post_wind_v": _F_slow_wind_v,
                        "wind_tau_u": _wind_tau_i_u,
                        "wind_tau_v": _wind_tau_j_v,
                        "wind_r1_rho0": _wind_r1_rho0,
                        "wind_r1_hu": _wind_r1_hu,
                        "wind_r1_hv": _wind_r1_hv,
                        "wind_increment_u": _wind_increment_u,
                        "wind_increment_v": _wind_increment_v,
                        "post_drag_u": _F_slow_drag_u,
                        "post_drag_v": _F_slow_drag_v,
                        "pre_external_u": F_slow_u,
                        "pre_external_v": F_slow_v,
                    },
                    (Hu_avg, Hv_avg),
                )
            state_new, (Hu_avg, Hv_avg) = _baro_result
            _stage_baro_override = (
                self._nemo_ws_test_hooks.stage_barotropic_output_override)
            if _stage_baro_override is not None:
                (_baro_eta, _baro_u, _baro_v,
                 Hu_avg, Hv_avg) = _stage_baro_override
                state_new = state_new._replace(
                    eta=state_new.eta.replace(data=_baro_eta),
                    uu_b=state_new.uu_b.replace(data=_baro_u),
                    vv_b=state_new.vv_b.replace(data=_baro_v),
                )

        _external_result_override = (
            self._nemo_ws_test_hooks.external_mode_result_override)
        if _external_result_override is not None:
            _eta_external, Hu_avg, Hv_avg = _external_result_override
            state_new = state_new._replace(
                eta=state_new.eta.replace(data=_eta_external))

        # NEMO-RK3 scheme identity: HYB is the live stprk3_stg barotropic
        # update (module default at :44; stages at :143-144,206-207,225), so
        # every Kaa stage receives the final external-mode velocity.  This is
        # the ONE WS stage ladder (stprk3_stg.F90:344-374): it runs for every
        # rk3_ws step, with the ACTUAL post-solve depth mean installed at every
        # Kaa, exactly where NEMO does at stprk3_stg.F90:433-446.  The private
        # ``stage_barotropic_correction`` hook ablates that external-mode
        # REPLACEMENT (below) for causal tests; it no longer selects between two
        # recurrences, because there is only one.  Public rk3_ws has no arm.
        # The stage-3 barotropic correction, deferred to its post-solve site
        # (stprk3_stg.F90:437-446 is BELOW the CALL dyn_zdf at :430).  None on
        # every path that does not run the WS stage ladder.
        _ws_stage3_correction = None
        if getattr(_cfg_b, "momentum_time_integrator", "euler") == "rk3_ws":
            # WEIGHTS of the stage depth-mean operator.  NEMO's is REFERENCE
            # weighted and fixed for the whole run:
            #     zub = uu_b(Kaa) - SUM( e3u_0(:)*uu(:,Kaa) ) * r1_hu_0
            #                                         stprk3_stg.F90:440
            #     hu_0 = SUM( e3u_0 * umask )          domain.F90:145
            # so M_ref(x) = SUM(e3u_0*x)/hu_0 and M_ref(umask) == 1 exactly.
            # Using the LIVE ``h_u_pre``/``H_u_pre`` instead is NOT the same
            # operator: ``h_u_pre`` is a MIN over two columns whose free-surface
            # Jacobians differ, so its per-level argmin can switch sides and the
            # live weights are not a uniform rescale of the reference ones.  On
            # a column whose two neighbours have equal depth the two coincide
            # (measured: one ULP), which is why this has no leverage on the flat
            # shelf and all of it on the staircase.
            # ``_ws_h_ref`` is the ssh=0 thickness, so ``min_cell_to_uface`` of
            # it, masked by the live 3-D face mask, IS ``e3u_0``, and its column
            # sum IS ``hu_0`` (both measured equal to NEMO's mesh_mask to 0.0
            # over every wet face).
            _ws_h_u_ref = min_cell_to_uface(_ws_h_ref) * _ws_u_live_mask
            _ws_h_v_ref = min_cell_to_vface(_ws_h_ref, _grid) * _ws_v_live_mask
            _ws_H_u_ref = jnp.maximum(jnp.sum(_ws_h_u_ref, axis=-1), 1e-10)
            _ws_H_v_ref = jnp.maximum(jnp.sum(_ws_h_v_ref, axis=-1), 1e-10)
            if self._nemo_ws_test_hooks.legacy_live_stage_mean_weights:
                _mean_h_u, _mean_H_u = h_u_pre, H_u_pre
                _mean_h_v, _mean_H_v = h_v_pre, H_v_pre
            else:
                _mean_h_u, _mean_H_u = _ws_h_u_ref, _ws_H_u_ref
                _mean_h_v, _mean_H_v = _ws_h_v_ref, _ws_H_v_ref
            if _return_live_stage_operands: _nemo_ws_live_baro_geometry = (_mean_h_u, _mean_h_v, nemo_reference_depth_reciprocal(_mean_H_u, state.u_mask.data), nemo_reference_depth_reciprocal(_mean_H_v, state.v_mask.data))  # noqa: E501,E701
            def _replace_stage_mean(u_in, v_in, target_u, target_v):
                if not self._nemo_ws_test_hooks.stage_barotropic_correction:
                    # Private causal arm: keep the stage's own depth mean.
                    return u_in * _ws_stage_u_mask, v_in * _ws_stage_v_mask
                # stprk3_stg.F90:440,444-445, in the shared barotropic home so
                # a fidelity gate can drive the SAME expression with NEMO's own
                # operands (a closure cannot be handed a record).
                # ROUND 36: NEMO MULTIPLIES by the reciprocal it stored once
                # (domain.f90:213), it does not divide by the depth, and its
                # SUM carries no mask -- the dry-column zero is inside the
                # reciprocal.  Built here from the same operands rather than
                # inside the operator so a fidelity gate can hand the operator
                # NEMO's OWN r1_hu_0 read from a record.
                return (
                    rk3_stage_barotropic_correction(
                        u_in, target_u, _mean_h_u,
                        nemo_reference_depth_reciprocal(
                            _mean_H_u, state.u_mask.data),
                        _ws_stage_u_mask),
                    rk3_stage_barotropic_correction(
                        v_in, target_v, _mean_h_v,
                        nemo_reference_depth_reciprocal(
                            _mean_H_v, state.v_mask.data),
                        _ws_stage_v_mask),
                )

            # NEMO's TARGET is not reconstructed from the 3-D velocity.  The
            # external mode writes its separately prognostic uu_b/vv_b(Kaa)
            # at dynspg_ts.F90:857-897 and every HYB stage reads that value
            # (stprk3_stg.F90:115-228,433-446).  Keep the old reduction only
            # for isolated non-card unit paths whose state has no NEMO pair;
            # all validated NEMO-identity cards carry it.
            if state_new.uu_b is not None and state_new.vv_b is not None:
                target_u = state_new.uu_b.data
                target_v = state_new.vv_b.data
            elif state_new.uu_b is None and state_new.vv_b is None:
                target_u = (jnp.sum(state_new.u.data * _mean_h_u, axis=-1)
                            / _mean_H_u * state.u_mask.data)
                target_v = (jnp.sum(state_new.v.data * _mean_h_v, axis=-1)
                            / _mean_H_v * state.v_mask.data)
            else:
                raise ValueError(
                    "NEMO prognostic depth mean requires both uu_b and vv_b")
            # stprk3_stg.f90:270 (the ``n_baro_upd = np_HYB`` branch the
            # compiled module's :48 default selects) divides the barotropic
            # transport by ``hu_0*(1+r3u(Kmm))`` -- the surface-height-ratio
            # column depth -- NOT by the sum of legoESM's min-rule face
            # thicknesses, which is first order wrong in the sea-surface
            # height difference ACROSS the face.  The same kernel already
            # supplies the advection's own divisor
            # (``momentum_flux_face_thickness``); it can supply this one
            # too.  LANDED round 206 (Decision 86).
            # SCOPE, stated because the code does not carry it: this target
            # is built ONCE per step from the step-entry (Kbb) sea surface,
            # while NEMO re-evaluates :270 inside every stage with that
            # stage's own ``r3u(Kmm)`` and legoESM's own stage transport
            # (``_nemo_ws_stage_transport``) does use the per-stage ssh.  So
            # stages 2 and 3 divide by the step-entry column depth here.
            # That placement is inherited from the structure this target
            # already had (``H_u_pre`` and ``Hu_avg`` are both step-entry
            # quantities); round 206 changed the depth RULE, not the time
            # level, and the time level is an open item.
            # FORM, also stated rather than implied: NEMO MULTIPLIES by a
            # stored reciprocal, ``un_adv*(r1_hu_0/(1+r3u))``, while this
            # DIVIDES by the summed depth.  Equal algebraically, not
            # bitwise.  The literal operand already exists -- the same
            # kernel returns ``r1_hu = r1_hu0/(1+r3u)`` under
            # ``include_reciprocals=True`` (vertical.py), which
            # ``_nemo_stage_corrected_velocity`` consumes through
            # ``nemo_source_round``.  Round 206 landed the measured
            # candidate the user approved, which is the divide; the
            # association is the next walk item, not an oversight.
            # Measured on VORTEX-zco: the two depths differ
            # by up to 0.129 m in 5000.86 m (2.58e-05 relative, 698 of 3660
            # u columns), and the stage-1 flux-form advection trend's
            # disagreement with NEMO falls from 6.285649e-11 to
            # 2.032879e-20 (u) with that divisor.  The private
            # ``"legacy_min_rule_depth"`` arm restores the old divisor as
            # the one-variable control.  This is the SHARED RK3 statement,
            # not a card option: NEMO's :48 ``n_baro_upd = np_HYB`` default
            # is compiled into every card's stprk3_stg, so every card that
            # runs the WS-RK3 transport reconcile takes it.
            _qco_tr_faces = _nemo_ws_qco_stage_faces(
                state.eta.data, _ws_h_ref, _ws_u_live_mask,
                _ws_v_live_mask, _grid)
            _H_u_transport = jnp.sum(_qco_tr_faces[0], axis=-1)
            _H_v_transport = jnp.sum(_qco_tr_faces[1], axis=-1)
            # Dry columns have a zero qco depth and carry no transport.
            _H_u_transport = jnp.where(_H_u_transport > 0.0,
                                       _H_u_transport, H_u_pre)
            _H_v_transport = jnp.where(_H_v_transport > 0.0,
                                       _H_v_transport, H_v_pre)
            transport_target_u = (
                Hu_avg / _H_u_transport * state.u_mask.data)
            transport_target_v = (
                Hv_avg / _H_v_transport * state.v_mask.data)
            def _transport_target_legacy():
                """The pre-round-206 target, for the private control arm.

                A function, not a value, so production traces NOTHING extra
                and the landed graph is the one the registry was measured
                on.  Identical to the two lines above when the landed
                statement is reverted -- which is the non-vacuity of the
                test that pins it.
                """
                return (Hu_avg / H_u_pre * state.u_mask.data,
                        Hv_avg / H_v_pre * state.v_mask.data)
            _use_transport_reconcile = (
                self._nemo_ws_test_hooks.momentum_transport_reconcile)
            _transport_target = (
                (transport_target_u, transport_target_v)
                if _use_transport_reconcile else None)

            # NEMO stprk3_stg.F90:257-304 builds ONE Kmm transport triplet
            # (zFu, zFv, zFw) per stage that dyn_adv (:315,331-334) and
            # tra_adv (:456-519) both consume, and :309-334,433-468
            # interleaves each barotropically corrected momentum stage with
            # its tracer stage.  So stage 2 evaluates eos+hpg on the stage-1
            # Kaa T/S/ssh, stage 3 on the stage-2 Kaa, and every stage RHS
            # carries dyn_adv_up3's vertical flux on that stage's explicit ww
            # (wAimp splits stage 3 only, :298-302).  The two hooks are
            # private one-variable controls; the public identity has no arm.
            _freeze_hpg = self._nemo_ws_test_hooks.freeze_stage_hpg_operands
            _omit_vert = self._nemo_ws_test_hooks.omit_stage_vertical_up3
            # ln_zad_Aimp=.false. is a legal NEMO namelist value.  With the
            # flag off, legoESM's ``tendencies()`` still carries the explicit
            # vertical momentum advection inside every stage RHS
            # (ocean_pe_latlon_cgrid: ``if not _aimp_vertadv: du_dt +=
            # diag_vertadv_u``), so the per-stage term below is Aimp-only --
            # exactly the gate the deleted once-per-step block had.  Adding it
            # here as well double-counted the term for Aimp=False + rk3_ws.
            _aimp_vertadv_ws = bool(
                getattr(_cfg_b, "adaptive_implicit_vertadv", False))
            _h_live_new = compute_layer_thickness(
                state_new.eta.data, state_new.H_bathy.data, _zc,
                min_water_column_m=_cfg_b.min_water_column_m)
            _h_live_one_third = h_k_old + (_h_live_new - h_k_old) / 3.0
            _h_live_one_half = 0.5 * (h_k_old + _h_live_new)
            _eta_live_one_third = (
                state.eta.data
                + (state_new.eta.data - state.eta.data) / 3.0)
            _eta_live_one_half = 0.5 * (
                state.eta.data + state_new.eta.data)
            _stage_entry_override = self._nemo_ws_test_hooks.stage_entry_override
            if _stage_entry_override is not None:
                _override_stage = _stage_entry_override[0]
                if _override_stage not in (2, 3):
                    raise ValueError(
                        "stage_entry_override stage must be 2 or 3")
                if _override_stage == 2:
                    _eta_live_one_third = _stage_entry_override[5]
                    _h_live_one_third = compute_layer_thickness(
                        _eta_live_one_third, state.H_bathy.data, _zc,
                        min_water_column_m=_cfg_b.min_water_column_m)
                else:
                    _eta_live_one_half = _stage_entry_override[5]
                    _h_live_one_half = compute_layer_thickness(
                        _eta_live_one_half, state.H_bathy.data, _zc,
                        min_water_column_m=_cfg_b.min_water_column_m)
            # Stage 3 has Kmm=N+1/2 (stprk3_stg.F90:218-235), and tra_zdf
            # consumes e3w(Kmm) in both off-diagonals (trazdf.F90:207-221).
            # The old whole-step entry eta is retained only as a private
            # one-variable ablation.
            _nemo_ws_zdf_eta_kmm = (
                state.eta.data
                if self._nemo_ws_test_hooks.legacy_zdf_entry_kmm_eta
                else _eta_live_one_half)
            if isinstance(_zc, OceanPartialCellCoordinate):
                _u_live_mask, _v_live_mask = compute_face_masks_3d(
                    _zc.is_active, _grid)
                _u_live_mask = _u_live_mask.astype(h_k_old.dtype)
                _v_live_mask = _v_live_mask.astype(h_k_old.dtype)
                _active_live = _zc.is_active.astype(h_k_old.dtype)
            else:
                _u_live_mask = state.u_mask.data[..., None]
                _v_live_mask = state.v_mask.data[..., None]
                _active_live = mask_3d
            # e3t_0: the reference (eta = 0) thickness ladder, so that
            # h_stage = e3t_0*(1 + eta/H) and sum_k e3t_0 = H_bathy.
            _h_ref_ws = compute_layer_thickness(
                jnp.zeros_like(state.eta.data), state.H_bathy.data, _zc,
                min_water_column_m=_cfg_b.min_water_column_m)
            _legacy_min_faces = (
                self._nemo_ws_test_hooks.legacy_stage_min_face_thickness)
            _legacy_reduced_transport_mean_arm = (
                self._nemo_ws_test_hooks
                .legacy_reduced_stage_transport_mean_arm)
            _stage_transport_kw = dict(
                h_ref=_h_ref_ws, Hu_avg=Hu_avg, Hv_avg=Hv_avg,
                u_mask_3d=_u_live_mask,
                v_mask_3d=_v_live_mask, grid=_grid, z_coord=_zc,
                H_bathy=state.H_bathy.data, config=_cfg_b,
                # RK3 div_hor applies the instantaneous river mass flux to
                # hdiv independently of the same runoff's external-mode SSH
                # forcing (sbcrnf.F90:253-260).  Preserve that distinct input.
                runoff_mass_flux=(
                    None if freshwater is None else freshwater.runoff),
                legacy_min_face_thickness=_legacy_min_faces,
                # wzv's Kbb/Kaa ssh operands (sshwzv.F90:334): the step-entry
                # level and the barotropic after-level, the same pair NEMO
                # passes at stprk3_stg.F90:297.
                eta_before=state.eta.data,
                eta_after=state_new.eta.data,
                # S-19 canonical option: GYRE/DINO resolve the NEMO literal
                # sshwzv recurrence.  The private hook remains the lane-1 arm
                # for cards whose resolved selector is generic.
                literal_wzv=(
                    getattr(_cfg_b, "wzv_call2_evaluation", "generic")
                    == "nemo_literal"
                    or self._nemo_ws_test_hooks.literal_stage_wzv),
                legacy_wzv_rederived_transport=(
                    self._nemo_ws_test_hooks
                    .legacy_wzv_rederived_transport),
                legacy_aimp_midpoint_w_metric=(
                    self._nemo_ws_test_hooks.legacy_aimp_midpoint_w_metric))
            # stprk3_stg.F90:365-388 has two live scheme arms.  The resolved
            # vector-invariant GYRE program advances stages 1/2 directly on
            # velocity (:366-369); only the flux-form arm uses the key_qco
            # (1+r3u) factors (:373-378).  STAGE 3 SELECTS THE SAME WAY: its
            # time step is inside dyn_zdf, and dynzdf.F90:119 carries the
            # identical IF( ln_dynadv_vec .OR. lk_linssh ), vector arm at
            # :121-122 and key_qco arm at :127-132.  (The sentence that used
            # to stand here said stage 3 enters the key_qco solve "in either
            # momentum program"; that was the transcription defect round 33
            # fixed, and all three stages now route through
            # rk3_stage_velocity_update.)
            def _qco_ratios(eta_stage):
                if self._nemo_ws_test_hooks.omit_stage_qco_factor:
                    return (
                        jnp.ones_like(state.u_mask.data)[..., None],
                        jnp.ones_like(state.v_mask.data)[..., None],
                    )
                _, _, _opu, _opv = _nemo_ws_qco_stage_faces(
                    eta_stage, _h_ref_ws, _u_live_mask, _v_live_mask, _grid)
                return _opu[..., None], _opv[..., None]

            _qu_b, _qv_b = _qco_ratios(state.eta.data)
            _qu_13, _qv_13 = _qco_ratios(_eta_live_one_third)
            _qu_12, _qv_12 = _qco_ratios(_eta_live_one_half)
            _qu_aa, _qv_aa = _qco_ratios(state_new.eta.data)
            from legoesm.ocean.eos import (
                nemo_r3t_rk3_stage1_stretch,
                nemo_r3t_stretch,
            )

            def _qco_t_ratio(eta_stage):
                return nemo_r3t_stretch(
                    _zc, eta_stage, state.H_bathy.data,
                    evaluation="nemo_reciprocal")

            _qt_b = _qco_t_ratio(state.eta.data)
            _qt_13 = nemo_r3t_rk3_stage1_stretch(
                _zc, state.eta.data, state_new.eta.data,
                state.H_bathy.data)
            _qt_12 = _qco_t_ratio(_eta_live_one_half)
            _qt_aa = _qco_t_ratio(state_new.eta.data)
            _tracer_qco_weights = (
                None
                if self._nemo_ws_test_hooks.legacy_tracer_stage_thickness_association
                else (
                    (_qt_b, _qt_b, _qt_13),
                    (_qt_b, _qt_13, _qt_12),
                    (_qt_b, _qt_12, _qt_aa),
                )
            )
            _vector_velocity_stage_update = (
                getattr(_cfg_b, "momentum_advection", "flux_form")
                == "vector_invariant"
                and not self._nemo_ws_test_hooks.legacy_vector_stage_qco_weights)
            # stprk3_stg.f90:356-360: the vector-invariant deck solves
            # continuity a SECOND time, on the raw stage velocity, for the
            # momentum vertical advection at every stage after the first.
            _momentum_wzv_split = nemo_stage_momentum_wzv_executes(
                _cfg_b, self._nemo_ws_test_hooks)
            _wall_live = (
                _active_live
                if getattr(_cfg_b, "tracer_wall_neumann_fill", True)
                else None)

            # tra_sbc_RK3 stages 1/2 (trasbc.F90:282-292): under nonlinear
            # free surface, add only the tracer carried by EMP, using the Kbb
            # surface concentration and Kmm top thickness.  In legoESM's sign
            # convention net_freshwater_flux = -NEMO emp, hence the positive
            # content source below.  Heat flux, salt flux, QSR, LDF and ZDF are
            # stage-3-only and remain in the later physics completion.
            _zero_stage_source = jnp.zeros_like(state.T.data)
            if freshwater is None:
                _emp_stage_mass_flux = jnp.zeros_like(state.eta.data)
            else:
                # The dilution operand is NEMO's ``emp`` ALONE
                # (trasbc.F90:282-288): the river runoff is NOT in it.  NEMO
                # carries the same runoff's WATER in two other places -- the
                # sea-surface forcing ``r1_rho0*(emp - rnf)``
                # (stp2d.F90:278-281) and the horizontal divergence
                # (sbcrnf.F90:279-283, called at divhor.F90:142) -- and the
                # TRACER it carries in a third, ``rnf_tsc`` (the river-runoff
                # block below).  legoESM's net freshwater is
                # ``precip - evap + runoff + ice_fw (+ restoring)``, which is
                # the right operand for the sea surface and the WRONG one
                # here: leaving the runoff in deposits a second, nearly
                # identical copy of the runoff's heat
                # (``rnf*T_top/rho0/h`` against ``MAX(sst,0)*rnf/rho0/h``).
                # The runoff is left OUT OF THE SUM rather than subtracted
                # from it: ``(x + r) - r`` is not bitwise ``x`` and on this
                # record the two spellings differ on 328 cells.  For a card
                # that resolves no runoff the two sums are bitwise equal,
                # because ``x + 0.0`` is ``x`` for every value this sum can
                # hold.
                _emp_stage_mass_flux = net_freshwater_flux(
                    freshwater, include_runoff=False)
            _stage_r1_rho0 = nemo_source_round(
                jnp.asarray(1.0, dtype=state.T.data.dtype)
                / jnp.asarray(_cfg_b.rho_0, dtype=state.T.data.dtype))

            def _emp_stage_rate(tracer, h_stage):
                # trasbc.f90:284-286 first stores r1_rho0/e3t(Kmm), then
                # multiplies EMP by the Kbb tracer and by that reciprocal.
                # ``_emp_stage_mass_flux`` is -EMP in legoESM's convention.
                z1_rho0_e3t = nemo_source_round(
                    _stage_r1_rho0
                    / jnp.maximum(h_stage[..., 0], 1.0e-10))
                top = nemo_source_round(
                    nemo_source_round(
                        _emp_stage_mass_flux * tracer[..., 0])
                    * z1_rho0_e3t) * _active_live[..., 0]
                return _zero_stage_source.at[..., 0].set(top)

            # River-runoff tracer source (trasbc.F90's river-runoff block).
            # It sits OUTSIDE the stage switch, so unlike the EMP/QNS block it
            # runs at ALL THREE stages, and NEMO forms the reciprocal of the
            # runoff depth FIRST (``zdep = 1/h_rnf``) and multiplies -- which
            # is not the same fp64 value as dividing.  With neither
            # ``ln_rnf_depth`` nor ``ln_rnf_depth_ini`` selected the runoff
            # depth is the LIVE top-cell thickness at the stage's Kmm and
            # ``nk_rnf = 1``, so the deposit is the top cell alone.
            _rnf_content = (
                None if surface_forcing is None
                else getattr(surface_forcing, "runoff_tracer_content", None))
            if _rnf_content is not None and len(_rnf_content) != 2:
                raise ValueError(
                    "surface_forcing.runoff_tracer_content must be the pair "
                    "(temperature content, salinity content) in NEMO rnf_tsc "
                    f"units; got {len(_rnf_content)} entries.")

            def _add_rnf_stage_rate(rate, content, h_stage):
                """``rate`` unchanged when the card carries no runoff.

                Returned unchanged rather than summed with a zero array: on a
                cell holding negative zero, ``-0.0 + 0.0`` is ``+0.0``, so a
                "harmless" addition is not bit-identical and GYRE's byte
                identity would turn on a signed zero.
                """
                if _rnf_content is None:
                    return rate
                zdep = 1.0 / jnp.maximum(h_stage[..., 0], 1.0e-10)
                top = jnp.asarray(content) * zdep * _active_live[..., 0]
                return rate.at[..., 0].add(top)

            def _stage_tracer_sources(h_stage):
                """EMP dilution then runoff, in the order NEMO accumulates."""
                rate_t = _emp_stage_rate(state.T.data, h_stage)
                rate_s = _emp_stage_rate(state.S.data, h_stage)
                if _rnf_content is None:
                    return rate_t, rate_s
                return (
                    _add_rnf_stage_rate(rate_t, _rnf_content[0], h_stage),
                    _add_rnf_stage_rate(rate_s, _rnf_content[1], h_stage),
                )

            def _stage_tracer_source_terms(h_stage):
                """Keep the two compiled statements separate until Krhs."""
                emp_t = _emp_stage_rate(state.T.data, h_stage)
                emp_s = _emp_stage_rate(state.S.data, h_stage)
                if _rnf_content is None:
                    return ((emp_t,), (emp_s,))
                zero_t = jnp.zeros_like(emp_t)
                zero_s = jnp.zeros_like(emp_s)
                return (
                    (emp_t,
                     _add_rnf_stage_rate(
                         zero_t, _rnf_content[0], h_stage)),
                    (emp_s,
                     _add_rnf_stage_rate(
                         zero_s, _rnf_content[1], h_stage)),
                )

            _stage3_T_rate = (
                tend.dT_dt.data * h_k_old
                / jnp.maximum(_h_live_one_half, 1.0e-10))
            _sw_pen_scheme_ws = getattr(
                getattr(
                    getattr(_cfg_b, "physics", None),
                    "shortwave_penetration", None),
                "scheme", None)
            if (
                _sw_pen_scheme_ws == "nemo_qsr_rgb"
                and surface_forcing is not None
                and getattr(surface_forcing, "sw_down", None) is not None
            ):
                # Same seam as the two-band arm below, for NEMO's three-band
                # chlorophyll penetration.  ``tra_qsr`` runs ONCE per step, at
                # stage 3, with Kmm (stprk3_stg.F90:581; traqsr.f90:213 ->
                # qsr_RGBc, whose live operands are ``e3t_0*(1+r3t(Kmm))`` and
                # ``gdepw_1d*(1+r3t(Kmm))``, traqsr.f90:388, :349).  The shared
                # pipeline already deposited the SAME kernel on the step-entry
                # (Kbb) ladder; rebuild that field with the pipeline's own two
                # operands so the subtraction below removes it exactly, then
                # add the Kmm evaluation.  No second RGB implementation.
                from legoesm.ocean.physics.shortwave_penetration import (
                    apply_shortwave_penetration,
                )
                _rgb_cfg = _cfg_b.physics.shortwave_penetration
                _rgb_chl = getattr(surface_forcing, "chl", None)
                if _rgb_chl is None:
                    raise ValueError(
                        "physics.shortwave_penetration.scheme='nemo_qsr_rgb' "
                        "needs a chlorophyll field; "
                        "OceanSurfaceForcing.chl is None")
                _rgb_dtype = h_k_old.dtype
                _rgb_sw = jnp.asarray(surface_forcing.sw_down, dtype=_rgb_dtype)
                _rgb_chl = jnp.asarray(_rgb_chl, dtype=_rgb_dtype)
                _rgb_gdepw_ref = -jnp.asarray(_zc.z_half_ref, dtype=_rgb_dtype)
                _rgb_e3t_ref = jnp.asarray(_zc.dz_ref, dtype=_rgb_dtype)
                # The pipeline's Kbb stretch is ``compute_ocean_jacobian`` on
                # the step-entry sea surface (physics/combined.py); use the
                # identical expression here so the two cancel bit for bit.
                _rgb_stretch_b = jnp.asarray(
                    compute_ocean_jacobian(
                        state.eta.data, state.H_bathy.data, _zc),
                    dtype=_rgb_dtype)
                _rgb_stretch_m = _h_live_one_half[..., 0] / jnp.maximum(
                    _h_ref_ws[..., 0], 1.0e-10)

                def _rgb_qsr(h_live, stretch):
                    return apply_shortwave_penetration(
                        _rgb_cfg, _rgb_sw,
                        chl=_rgb_chl,
                        dz_live=h_live,
                        wet_cell=jnp.asarray(h_live > 0.0, dtype=_rgb_dtype),
                        gdepw_bottom_live=(
                            _rgb_gdepw_ref[1:] * stretch[..., jnp.newaxis]),
                        gdepw_ref=_rgb_gdepw_ref,
                        e3t_ref=_rgb_e3t_ref,
                        rho_0=_cfg_b.rho_0,
                        c_sw=_cfg_b.physics.constants.c_sw,
                    )

                _stage3_T_rate = _nemo_qsr_stage3_rate(
                    tend.dT_dt.data,
                    _rgb_qsr(h_k_old, _rgb_stretch_b),
                    _rgb_qsr(_h_live_one_half, _rgb_stretch_m),
                    h_k_old, _h_live_one_half)
            elif (
                _sw_pen_scheme_ws == "nemo_qsr_2bd"
                and surface_forcing is not None
                and getattr(surface_forcing, "sw_down", None) is not None
            ):
                # The shared physics pipeline evaluates its two-band kernel on
                # Kbb.  qsr_2BD is called at stage 3 with gdepw/e3t(Kmm)
                # (stprk3_stg.F90:581; traqsr.F90:665-712).  Replace only that
                # already-present component with the same shared kernel on the
                # live half-stage ladder; no second shortwave implementation.
                from legoesm.ocean.physics.shortwave_penetration import (
                    shortwave_penetration_tendency,
                )
                _r3t_b = h_k_old[..., 0] / jnp.maximum(
                    _h_ref_ws[..., 0], 1.0e-10)
                _r3t_m = _h_live_one_half[..., 0] / jnp.maximum(
                    _h_ref_ws[..., 0], 1.0e-10)
                _qsr_stretch_override = (
                    self._nemo_ws_test_hooks.stage3_qsr_stretch_override)
                if _qsr_stretch_override is not None:
                    _r3t_m = _qsr_stretch_override
                _qsr_b = shortwave_penetration_tendency(
                    surface_forcing.sw_down,
                    _zc.dz_ref, _zc.z_half_ref, _r3t_b,
                    _cfg_b.physics.shortwave_penetration,
                    rho_0=_cfg_b.rho_0,
                    c_sw=_cfg_b.physics.constants.c_sw,
                    z_half_stretch=_r3t_b,
                )
                _qsr_m = shortwave_penetration_tendency(
                    surface_forcing.sw_down,
                    _zc.dz_ref, _zc.z_half_ref, _r3t_m,
                    _cfg_b.physics.shortwave_penetration,
                    rho_0=_cfg_b.rho_0,
                    c_sw=_cfg_b.physics.constants.c_sw,
                    z_half_stretch=_r3t_m,
                )
                _stage3_T_rate = _nemo_qsr_stage3_rate(
                    tend.dT_dt.data, _qsr_b, _qsr_m,
                    h_k_old, _h_live_one_half)

            _stage12_source_terms = (
                None
                if _rnf_content is None
                else (
                    _stage_tracer_source_terms(h_k_old),
                    _stage_tracer_source_terms(_h_live_one_third),
                    None,
                )
            )
            _stage_source_rates = (
                _stage_tracer_sources(h_k_old),
                _stage_tracer_sources(_h_live_one_third),
                # stprk3_stg.F90:565-600 clears Krhs, accumulates advection,
                # nonlinear-SBC, QSR and LDF, then tra_zdf.F90 combines that
                # stage-3 rate with Kbb using the Kmm QCO weight.  These rates
                # must therefore enter the stage-3 RHS -- pre-applying them to
                # the Kbb tracer base changes their coefficient whenever
                # r3t(Kbb) != r3t(Kmm), which is live in GYRE.
                # ``tendencies`` expressed the physical fluxes as a
                # concentration rate using the step-entry thickness.  NEMO
                # tra_sbc_RK3/tra_qsr divide the same flux by e3t(Kmm)
                # (trasbc.F90:299-315; traqsr.F90:665-712).  Convert through
                # content here so the stage helper's h_rhs multiplication
                # cancels the Kmm divisor, rather than spuriously weighting
                # surface heat by h(Kmm)/h(Kbb).
                (
                    _add_rnf_stage_rate(
                        _stage3_T_rate,
                        None if _rnf_content is None else _rnf_content[0],
                        _h_live_one_half),
                    _add_rnf_stage_rate(
                        tend.dS_dt.data * h_k_old
                        / jnp.maximum(_h_live_one_half, 1.0e-10),
                        None if _rnf_content is None else _rnf_content[1],
                        _h_live_one_half),
                ),
            )
            if _return_tracer_process_trace:
                # Round-124 WRITE-only process budget.  Materialize the two
                # source components inside the full production step; the
                # carried trajectory comes from a separate ordinary compiled
                # call in ``step`` below.  ``tend.dT_dt`` contains the shared
                # pipeline's Kbb qsr_2BD field plus the external qns surface
                # deposit.  `_qsr_b` is the NEMO-live Kbb reconstruction used
                # by the stage-3 replacement and need not be bit-identical to
                # the pipeline field.  Re-read the SAME physics callable with
                # production's cell-centred proxy so the surface row removes
                # the component actually accumulated into `tend`; the entire
                # remainder stays with QSR.
                _process_cc_state = state._replace(
                    u=state.u.replace(data=0.5 * (
                        state.u.data[:, :-1, :] + state.u.data[:, 1:, :])),
                    v=state.v.replace(data=0.5 * (
                        state.v.data[:-1, :, :] + state.v.data[1:, :, :])),
                )
                _process_physics = self._physics_fn(
                    _process_cc_state, _grid, _zc, surface_forcing)
                _process_qsr_kbb = _process_physics.dT_dt.data
                _nemo_ws_process_surface_rate = (
                    (tend.dT_dt.data - _process_qsr_kbb) * h_k_old
                    / jnp.maximum(_h_live_one_half, 1.0e-10))
                _nemo_ws_process_qsr_rate = (
                    _stage3_T_rate - _nemo_ws_process_surface_rate)
                _process_plant = self._nemo_ws_test_hooks.tracer_process_trace
                if _process_plant:
                    _pj, _pi, _pk, _pdelta = _process_plant
                    _nemo_ws_process_surface_rate = (
                        _nemo_ws_process_surface_rate.at[
                            _pj, _pi, _pk].add(_pdelta))
                    _stage3_T_rate = _stage3_T_rate.at[
                        _pj, _pi, _pk].add(_pdelta)
                    _stage_source_rates = (
                        *_stage_source_rates[:2],
                        (_stage3_T_rate, _stage_source_rates[2][1]))
                _nemo_ws_process_qco = (
                    _qt_b, _qt_12, _qt_aa,
                    h_k_old, _h_live_one_half, _h_live_new)
                _nemo_ws_qsr_association = _NEMOWSQsrAssociationTrace(
                    tendency_kbb=tend.dT_dt.data,
                    qsr_kbb=_qsr_b,
                    qsr_kmm=_qsr_m,
                    thickness_kbb=h_k_old,
                    thickness_kmm=_h_live_one_half,
                    process_qsr_kbb=_process_qsr_kbb,
                    process_surface_rate=_nemo_ws_process_surface_rate,
                    process_qsr_rate=_nemo_ws_process_qsr_rate,
                )

            def _momentum_stage_w(geom):
                # NEMO's momentum consumers read the field its OWN continuity
                # solve produced (stprk3_stg.f90:360); the tracer transport
                # re-solves and overwrites it (traadv.f90:274).  Slot 11 is
                # None on cards that do not run the first solve, and then both
                # consumers read the one field legoESM builds, as before.
                return geom[2] if geom[11] is None else geom[11]

            def _stage_vertical_up3(u_stage, v_stage, geom):
                # dynadv_up3.F90:239-358: vertical flux of the stage Kmm
                # velocity on the stage transport's explicit ww, divided by
                # e3u(Kmm).  Identically zero from rest (kt=1 stage 1).
                if _omit_vert or not _aimp_vertadv_ws:
                    return None
                w_mom = _momentum_stage_w(geom)
                return (
                    nemo_up3_vertical_momentum_advection(
                        u_stage * u_mask_3d, interp_cell_to_uface(w_mom),
                        geom[4], face_active=_u_live_mask),
                    nemo_up3_vertical_momentum_advection(
                        v_stage * v_mask_3d,
                        interp_cell_to_vface(w_mom, _grid),
                        geom[5], face_active=_v_live_mask),
                )

            def _stage_tracers(stop_after_stage, stage_geometry, resume=None):
                # Same base as the tracer program's T_mid (state_new.T is the
                # post-physics-Euler T_new from step 2), same Kmm transports.
                # NEMO advances ts(Kbb) by advection (+sbc) only at stages 1-2
                # and adds every physics term at stage 3 (stprk3_stg.F90:
                # 519-552 vs :556-600); legoESM's physics-Euler-then-stages
                # ordering predates this round and is inert on the certified
                # cards (K_h = K_v = 0, no forcing).
                return _nemo_ws_rk3_tracer_pair_step(
                    state.T.data, state.S.data,
                    _cfg_b.tracer_advection,
                    _g0[0], _g0[1], _g0[2], h_k_old, _h_live_new,
                    _g0[4], _g0[5], _grid, dt, _active_live,
                    recon_fill_mask=_wall_live,
                    linssh_top_flux=getattr(_zc, "linear_free_surface", False),
                    stage_transport_geometry=stage_geometry,
                    stage_source_rates=_stage_source_rates,
                    stage_source_terms=_stage12_source_terms,
                    stage_qco_weights=_tracer_qco_weights,
                    stop_after_stage=stop_after_stage,
                    resume=resume,
                    return_stage1_trace=(
                        stop_after_stage == 1
                        and bool(self._nemo_ws_test_hooks
                                 .expose_tracer_stage1_boundary)),
                )

            def _tracer_transport_geometry_override(geometry, override):
                """Map a private owned zF triplet into the shared geometry."""
                if override is None:
                    return geometry
                zfu_owned, zfv_owned, zfw = override
                zfu = jnp.concatenate([zfu_owned[:, -1:, :], zfu_owned], axis=1)
                zfv = jnp.concatenate([zfv_owned[-1:, :, :], zfv_owned], axis=0)
                return (
                    zfu / jnp.asarray(_grid.dy_u)[..., None],
                    zfv / jnp.asarray(_grid.dx_v)[..., None],
                    zfw / jnp.asarray(_grid.area_T)[..., None],
                    geometry[3], geometry[4], geometry[5], geometry[6],
                    zfu, zfv, geometry[9], geometry[10],
                    *geometry[11:],
                )

            def _stage_hpg_operands(T_stage, S_stage, eta_stage):
                # Operands of the stage-2/3 eos+dyn_hpg call.  ``None`` keeps
                # the step-entry bundle (Kbb operands, harness Arm A control);
                # the two split hooks freeze one operand class at a time.
                if _freeze_hpg:
                    return None
                if self._nemo_ws_test_hooks.freeze_stage_hpg_tracers:
                    T_stage, S_stage = state.T.data, state.S.data
                if self._nemo_ws_test_hooks.freeze_stage_hpg_eta:
                    eta_stage = state.eta.data
                return (T_stage, S_stage, eta_stage)

            def _stage2_hpg_operands(T_stage, S_stage, eta_stage):
                override = (
                    self._nemo_ws_test_hooks.stage2_thermodynamic_override)
                if override is not None:
                    return override
                return _stage_hpg_operands(T_stage, S_stage, eta_stage)

            # stage 1 (dt/3): Kmm = Kbb transport, full RHS incl. vertical UP3
            _g0 = _nemo_ws_stage_transport(
                (u0, v0), h_k_old, 0, eta_stage=state.eta.data,
                dt=(dt / 3.0 if self._nemo_ws_test_hooks.source_stage_wzv_clock_arm
                    else dt),
                barotropic_velocity=(
                    None if _legacy_reduced_transport_mean_arm else
                    _nemo_ws_stage_barotropic_velocity(
                        1,
                        ((state.uu_b.data, state.vv_b.data)
                         if state.uu_b is not None and state.vv_b is not None
                         else (jnp.zeros_like(target_u),
                               jnp.zeros_like(target_v))),
                        (target_u, target_v))),
                **_stage_transport_kw)
            if self._nemo_ws_test_hooks.expose_stage1_wzv:
                _nemo_ws_exposed_stage1_wzv = _g0[2]
            _operand_name = (
                self._nemo_ws_test_hooks.expose_stage1_transport_operand)
            if _operand_name == "thickness":
                _nemo_ws_exposed_stage1_transport_operand = (_g0[4], _g0[5])
            elif _operand_name == "corrected_velocity":
                _nemo_ws_exposed_stage1_transport_operand = (_g0[9], _g0[10])
            elif _operand_name == "transport_average":
                _nemo_ws_exposed_stage1_transport_operand = (
                    jnp.broadcast_to(Hu_avg[..., None], _g0[9].shape),
                    jnp.broadcast_to(Hv_avg[..., None], _g0[10].shape))
            elif _operand_name:
                raise ValueError(
                    "expose_stage1_transport_operand must be empty, "
                    "'thickness', 'corrected_velocity', or "
                    "'transport_average'")
            # The compiled vector program completes Krhs in stp_2D and stage
            # 1 consumes it directly: the transport/W built above feeds the
            # tracer path, while only the non-vector branch calls dyn_adv
            # here (stprk3_stg.F90:326-374,661-675).  Keep the projected,
            # post-external recomputation solely for that non-vector arm.
            _vert0 = (None if _vector_velocity_stage_update else
                      _stage_vertical_up3(u0, v0, _g0))
            _du1_rhs, _dv1_rhs = (
                (du_dt, dv_dt) if _vector_velocity_stage_update
                else (du_dt_pert, dv_dt_pert))
            _nemo_ws_live_stage1_full_rhs = (du_dt, dv_dt)
            _stage1_rhs_base = (_du1_rhs, _dv1_rhs)
            # The stage's NEMO e3u/e3v(Kmm) pair for the flux-form momentum
            # advection, from the ONE kernel (_nemo_ws_qco_stage_faces) keyed
            # on the stage ssh -- NOT read off the stage transport's geom[4],
            # so the older ``legacy_stage_min_face_thickness`` transport arm
            # stays one-variable (review finding).  The private hook hands
            # None so tendencies() falls back to its own min-rule thickness.
            _legacy_hadv_h = (
                self._nemo_ws_test_hooks.legacy_hadv_min_face_thickness)

            def _stage_face_thickness(eta_stage):
                if _legacy_hadv_h:
                    return None
                return _nemo_ws_qco_stage_faces(
                    eta_stage, _h_ref_ws, _u_live_mask, _v_live_mask, _grid)[:2]

            def _stage_zad_operands(stage, live):
                # Rounds 158/194.  Replace only the slots the private hook
                # names, so one dyn_zad operand at a time can come from NEMO's
                # recorded stage while all other stage inputs stay legoESM's.
                overrides = (
                    self._nemo_ws_test_hooks.stage2_zad_operand_override,
                    self._nemo_ws_test_hooks.stage3_zad_operand_override)
                override = overrides[stage - 2]
                if override is None:
                    return live
                return tuple(
                    live[index] if override[index] is None
                    else override[index] for index in range(3))

            _face_thickness_kbb = (
                None if _vector_velocity_stage_update
                else _stage_face_thickness(state.eta.data))
            if (_transport_target is not None
                    and not _vector_velocity_stage_update):
                _p0_with_zub = _mom_pert_ws(
                    u0, v0, False, _transport_target,
                    stage_face_thickness=_face_thickness_kbb, stage_index=1)
                _p0_no_zub = _mom_pert_ws(
                    u0, v0, False, None,
                    stage_face_thickness=_face_thickness_kbb, stage_index=1)
                _du1_rhs = _du1_rhs + (_p0_with_zub[0] - _p0_no_zub[0])
                _dv1_rhs = _dv1_rhs + (_p0_with_zub[1] - _p0_no_zub[1])
            _stage1_rhs_post_transport = (_du1_rhs, _dv1_rhs)
            if not _vector_velocity_stage_update:
                # In the non-vector program, replace the pre-external ZAD
                # association with the stage transport's W operand.
                _p0_with_zad = _mom_pert_ws(
                    u0, v0, False, None,
                    stage_face_thickness=_face_thickness_kbb,
                    stage_zad_operands=(_g0[2], _g0[4], _g0[5]),
                    stage_index=1)
                _p0_without_zad = _mom_pert_ws(
                    u0, v0, False, None,
                    stage_face_thickness=_face_thickness_kbb, stage_index=1)
                _du1_rhs = _du1_rhs + (_p0_with_zad[0] - _p0_without_zad[0])
                _dv1_rhs = _dv1_rhs + (_p0_with_zad[1] - _p0_without_zad[1])
            _stage1_rhs_post_zad = (_du1_rhs, _dv1_rhs)
            # Stage 1: Kmm = Kbb, so the RHS carries (1 + r3u(Kbb)).
            _u1_rhs = _du1_rhs if _vert0 is None else _du1_rhs + _vert0[0]
            _v1_rhs = _dv1_rhs if _vert0 is None else _dv1_rhs + _vert0[1]
            _nemo_ws_live_stage1_rhs_walk = (
                _stage1_rhs_base,
                _stage1_rhs_post_transport,
                _stage1_rhs_post_zad,
                (_u1_rhs, _v1_rhs),
            )
            if self._nemo_ws_test_hooks.expose_stage1_momentum_rhs:
                _nemo_ws_exposed_stage1_rhs = (_u1_rhs, _v1_rhs)
            _stage1_split_arm = (
                self._nemo_ws_test_hooks.expose_stage1_momentum_rhs_split)
            if _stage1_split_arm:
                if _nemo_ws_stage1_main_advection is None:
                    raise ValueError(
                        "expose_stage1_momentum_rhs_split needs the "
                        "step-level per-term decomposition")
                if _aimp_vertadv_ws:
                    # With ln_zad_Aimp=.true. the step-level tendency does
                    # NOT carry the vertical advection that ``advection_u``
                    # reports (it is applied as its own operator-split
                    # stage), so subtracting the reported half would remove a
                    # term the array never held.  Fail closed.
                    raise ValueError(
                        "expose_stage1_momentum_rhs_split is defined only "
                        "with adaptive_implicit_vertadv disabled")
                if _stage1_split_arm == "completed":
                    _nemo_ws_exposed_stage1_rhs = (_u1_rhs, _v1_rhs)
                elif _stage1_split_arm in ("advection_horizontal",
                                           "advection_vertical",
                                           "advection_zub_increment"):
                    # The two halves of the SAME removed content, published
                    # apart so ``dyn_adv_up3``'s horizontal flux divergence
                    # (dynadv_up3.f90:174-215) and its vertical block
                    # (:245-360) can be scored part by part.  Each stage
                    # increment belongs to exactly one half: the ``zub``
                    # transport increment reaches the flux-form horizontal
                    # term alone, the stage ZAD increment reaches dyn_zad's
                    # operands alone.
                    _hpart, _vpart = _nemo_ws_stage1_advection_halves
                    _dt_u = (_stage1_rhs_post_transport[0]
                             - _stage1_rhs_base[0])
                    _dt_v = (_stage1_rhs_post_transport[1]
                             - _stage1_rhs_base[1])
                    _dz_u = (_stage1_rhs_post_zad[0]
                             - _stage1_rhs_post_transport[0])
                    _dz_v = (_stage1_rhs_post_zad[1]
                             - _stage1_rhs_post_transport[1])
                    if _stage1_split_arm == "advection_zub_increment":
                        # The horizontal half's barotropic cross-term on its
                        # own: what replacing the raw Kmm velocity by the
                        # zub-corrected transport velocity
                        # (stprk3_stg.f90:264-277) does to the trend.
                        _nemo_ws_exposed_stage1_rhs = (_dt_u, _dt_v)
                    elif _stage1_split_arm == "advection_horizontal":
                        _nemo_ws_exposed_stage1_rhs = (
                            _hpart[0] + _dt_u, _hpart[1] + _dt_v)
                    else:
                        _nemo_ws_exposed_stage1_rhs = (
                            _vpart[0] + _dz_u, _vpart[1] + _dz_v)
                else:
                    # ``_stage1_rhs_base`` is the stage-1 right-hand side
                    # before the three addends that follow it, and ALL THREE
                    # are pure advection: the zub transport operand, the
                    # stage ZAD operand, and the explicit vertical UP3 term
                    # ``_vert0`` -- which is None here because the guard
                    # above refuses this hook whenever the adaptive implicit
                    # vertical advection that would make it non-None is on.
                    # So removing the step-level advection component leaves
                    # exactly the non-advective right-hand side: NEMO's
                    # ``Krhs`` as ``stp_2D`` hands it to stage 1.
                    _nemo_ws_exposed_stage1_rhs = (
                        _stage1_rhs_base[0]
                        - _nemo_ws_stage1_main_advection[0],
                        _stage1_rhs_base[1]
                        - _nemo_ws_stage1_main_advection[1])
            if self._nemo_ws_test_hooks.stage1_momentum_rhs_override is not None:
                _u1_rhs, _v1_rhs = (
                    self._nemo_ws_test_hooks.stage1_momentum_rhs_override)
            u1_raw = rk3_stage_velocity_update(
                u0, _u1_rhs, dt_mom / 3.0, _ws_stage_u_mask,
                vector_form=_vector_velocity_stage_update,
                qco_before=_qu_b, qco_now=_qu_b, qco_after=_qu_13)
            v1_raw = rk3_stage_velocity_update(
                v0, _v1_rhs, dt_mom / 3.0, _ws_stage_v_mask,
                vector_form=_vector_velocity_stage_update,
                qco_before=_qv_b, qco_now=_qv_b, qco_after=_qv_13)
            if self._nemo_ws_test_hooks.expose_stage1_raw_momentum:
                _nemo_ws_exposed_stage1_raw = (u1_raw, v1_raw)
            u1_corr, v1_corr = _replace_stage_mean(
                u1_raw, v1_raw, target_u, target_v)
            _g0_tracer = _tracer_transport_geometry_override(
                _g0,
                self._nemo_ws_test_hooks.stage1_tracer_transport_override)
            _stage1_tracer_result = _stage_tracers(
                1, (_g0_tracer, _g0, _g0))
            if self._nemo_ws_test_hooks.expose_tracer_stage1_boundary:
                (
                    _T_stage1, _S_stage1,
                    _T_stage1_after_adv, _S_stage1_after_adv,
                    _T_stage1_after_sbc, _S_stage1_after_sbc,
                ) = _stage1_tracer_result
                if (self._nemo_ws_test_hooks.expose_tracer_stage1_boundary
                        == "after_advection"):
                    _nemo_ws_exposed_tracer_boundary = (
                        _T_stage1_after_adv, _S_stage1_after_adv)
                else:
                    _nemo_ws_exposed_tracer_boundary = (
                        _T_stage1_after_sbc, _S_stage1_after_sbc)
            else:
                _T_stage1, _S_stage1 = _stage1_tracer_result
            if (_stage_entry_override is not None
                    and _stage_entry_override[0] == 2):
                (_, u1_corr, v1_corr, _T_stage1, _S_stage1,
                 _eta_live_one_third) = _stage_entry_override
            if self._nemo_ws_test_hooks.expose_tracer_stage == 1:
                _nemo_ws_exposed_tracer_stage = (
                    _T_stage1, _S_stage1, _eta_live_one_third)
            # stage 2 (dt/2): Kmm = stage-1 Kaa
            _g1 = _nemo_ws_stage_transport(
                (u1_corr, v1_corr), _h_live_one_third, 1,
                eta_stage=_eta_live_one_third,
                dt=(dt / 2.0 if self._nemo_ws_test_hooks.source_stage_wzv_clock_arm
                    else dt),
                barotropic_velocity=(
                    None if _legacy_reduced_transport_mean_arm else
                    _nemo_ws_stage_barotropic_velocity(
                        2,
                        ((state.uu_b.data, state.vv_b.data)
                         if state.uu_b is not None and state.vv_b is not None
                         else (jnp.zeros_like(target_u),
                               jnp.zeros_like(target_v))),
                        (target_u, target_v))),
                velocity_form_wzv=(
                    self._nemo_ws_test_hooks.stage2_wzv_velocity_form),
                momentum_velocity_form_w=_momentum_wzv_split,
                momentum_wzv_clock=(
                    (dt / 2.0,
                     0.5 * (state.eta.data + state_new.eta.data))
                    if self._nemo_ws_test_hooks
                    .stage2_momentum_wzv_clock_pair else None),
                **_stage_transport_kw)
            p1u_corr, p1v_corr = _mom_pert_ws(
                u1_corr, v1_corr, True, _transport_target,
                _stage2_hpg_operands(
                    _T_stage1, _S_stage1, _eta_live_one_third),
                _stage_vertical_up3(u1_corr, v1_corr, _g1),
                stage_face_thickness=_stage_face_thickness(_eta_live_one_third),
                stage_zad_operands=_stage_zad_operands(2,
                    (_momentum_stage_w(_g1), _g1[4], _g1[5])),
                stage_index=2)
            _stage2_rhs_production = (p1u_corr, p1v_corr)
            if self._nemo_ws_test_hooks.stage2_momentum_rhs_override is not None:
                p1u_corr, p1v_corr = (
                    self._nemo_ws_test_hooks.stage2_momentum_rhs_override)
            if self._nemo_ws_test_hooks.expose_stage2_momentum_rhs:
                _nemo_ws_exposed_stage2_rhs = _stage2_rhs_production
            u2_raw = rk3_stage_velocity_update(
                u0, p1u_corr, dt_mom / 2.0, _ws_stage_u_mask,
                vector_form=_vector_velocity_stage_update,
                qco_before=_qu_b, qco_now=_qu_13, qco_after=_qu_12)
            v2_raw = rk3_stage_velocity_update(
                v0, p1v_corr, dt_mom / 2.0, _ws_stage_v_mask,
                vector_form=_vector_velocity_stage_update,
                qco_before=_qv_b, qco_now=_qv_13, qco_after=_qv_12)
            if self._nemo_ws_test_hooks.expose_stage2_raw_momentum:
                _nemo_ws_exposed_stage2_raw = (u2_raw, v2_raw)
            u2_corr, v2_corr = _replace_stage_mean(
                u2_raw, v2_raw, target_u, target_v)
            _g1_tracer = _tracer_transport_geometry_override(
                _g1,
                self._nemo_ws_test_hooks.stage2_tracer_transport_override)
            _T_stage2, _S_stage2 = _stage_tracers(
                2, (_g0, _g1_tracer, _g1),
                resume=(1, _T_stage1, _S_stage1))
            if (_stage_entry_override is not None
                    and _stage_entry_override[0] == 3):
                (_, u2_corr, v2_corr, _T_stage2, _S_stage2,
                 _eta_live_one_half) = _stage_entry_override
            if self._nemo_ws_test_hooks.expose_tracer_stage == 2:
                _nemo_ws_exposed_tracer_stage = (
                    _T_stage2, _S_stage2, _eta_live_one_half)
            _nemo_ws_stage_tracers = (_T_stage2, _S_stage2)
            # stage 3 (dt): Kmm = stage-2 Kaa; ww split into explicit/implicit
            _g2 = _nemo_ws_stage_transport(
                (u2_corr, v2_corr), _h_live_one_half, 2,
                eta_stage=_eta_live_one_half,
                dt=dt,
                barotropic_velocity=(
                    None if _legacy_reduced_transport_mean_arm else
                    _nemo_ws_stage_barotropic_velocity(
                        3,
                        ((state.uu_b.data, state.vv_b.data)
                         if state.uu_b is not None and state.vv_b is not None
                         else (jnp.zeros_like(target_u),
                               jnp.zeros_like(target_v))),
                        (target_u, target_v))),
                momentum_velocity_form_w=_momentum_wzv_split,
                **_stage_transport_kw)
            _g2_override = self._nemo_ws_test_hooks.stage3_transport_override
            _g2 = _tracer_transport_geometry_override(_g2, _g2_override)
            if _g2_override is not None:
                _g2 = (*_g2[:6], jnp.zeros_like(_g2[6]), *_g2[7:])
            _stage3_hpg_operands = _stage_hpg_operands(
                _T_stage2, _S_stage2, _eta_live_one_half)
            _stage3_vertical_up3 = _stage_vertical_up3(u2_corr, v2_corr, _g2)
            _stage3_face_thickness = _stage_face_thickness(_eta_live_one_half)
            p2u_corr, p2v_corr = _mom_pert_ws(
                u2_corr, v2_corr, False, _transport_target,
                _stage3_hpg_operands,
                _stage3_vertical_up3,
                stage_face_thickness=_stage3_face_thickness,
                stage_zad_operands=_stage_zad_operands(3, (
                    _momentum_stage_w(_g2), _g2[4], _g2[5])),
                stage_index=3)
            _expose_stage3_rhs = (
                self._nemo_ws_test_hooks.expose_stage3_momentum_rhs)
            if _expose_stage3_rhs == "post_ldf":
                # stprk3_stg.F90:430 -- the operand dyn_zdf receives.
                _nemo_ws_exposed_stage3_rhs = (p2u_corr, p2v_corr)
            elif _expose_stage3_rhs == "pre_ldf":
                # stprk3_stg.F90:400 -- the operand dyn_ldf receives.  A second
                # WRITE-only evaluation of the SAME stage with the lateral
                # term withheld; the production RHS above is untouched.
                _nemo_ws_exposed_stage3_rhs = _mom_pert_ws(
                    u2_corr, v2_corr, True, _transport_target,
                    _stage3_hpg_operands,
                    _stage3_vertical_up3,
                    stage_face_thickness=_stage3_face_thickness,
                    stage_zad_operands=_stage_zad_operands(3, (
                    _momentum_stage_w(_g2), _g2[4], _g2[5])),
                    stage_index=3)
            elif _expose_stage3_rhs:
                raise ValueError(
                    "expose_stage3_momentum_rhs must be empty, 'pre_ldf', or "
                    f"'post_ldf'; got {_expose_stage3_rhs!r}")
            # ARM.  The stage-3 time step is dyn_zdf's, and dynzdf.F90:119
            # selects on the SAME ln_dynadv_vec .OR. lk_linssh predicate that
            # stprk3_stg.F90:365 uses at stages 1 and 2 -- vector arm at
            # dynzdf.F90:121-122, key_qco arm at :127-132.  Same helper, same
            # selector, so stage 3 cannot drift from stages 1 and 2 again.
            u3_raw = rk3_stage_velocity_update(
                u0, p2u_corr, dt_mom, _ws_stage_u_mask,
                vector_form=_vector_velocity_stage_update,
                qco_before=_qu_b, qco_now=_qu_12, qco_after=_qu_aa)
            v3_raw = rk3_stage_velocity_update(
                v0, p2v_corr, dt_mom, _ws_stage_v_mask,
                vector_form=_vector_velocity_stage_update,
                qco_before=_qv_b, qco_now=_qv_12, qco_after=_qv_aa)
            # ORDER.  The barotropic correction (stprk3_stg.F90:437-446) sits
            # BELOW the CALL dyn_zdf at stprk3_stg.F90:430, so at stage 3 it
            # runs AFTER the implicit vertical solve, never before it.  Stage
            # 3 therefore hands the solve the masked explicit update -- the
            # vector dyn_zdf builds for itself at dynzdf.F90:121-122 -- and
            # ``_ws_stage3_correction`` below applies the correction ONCE, at
            # the post-solve site.  Stages 1 and 2 keep their inline call:
            # NEMO's banner at stprk3_stg.F90:433 says "All stages", and no
            # solve runs between a stage update and its correction there.
            u3_corr = u3_raw * _ws_stage_u_mask
            v3_corr = v3_raw * _ws_stage_v_mask
            _ws_stage3_correction = (
                _replace_stage_mean, target_u, target_v)
            _nemo_ws_live_stage_geometry = (_g0, _g1, _g2)
            _nemo_ws_live_stage_states = (
                (u0, v0, state.T.data, state.S.data, state.eta.data),
                (u1_corr, v1_corr, _T_stage1, _S_stage1,
                 _eta_live_one_third),
                (u2_corr, v2_corr, _T_stage2, _S_stage2,
                 _eta_live_one_half),
            ); _nemo_ws_live_stage_raw = ((u1_raw, v1_raw), (u2_raw, v2_raw), (u3_raw, v3_raw)); _nemo_ws_live_stage_rhs = ((_u1_rhs, _v1_rhs), (p1u_corr, p1v_corr), (p2u_corr, p2v_corr))  # noqa: E501,E702
            _nemo_ws_live_stage_qco = (
                (_qt_b - 1.0, _qu_b - 1.0, _qv_b - 1.0),
                (_qt_13 - 1.0, _qu_13 - 1.0, _qv_13 - 1.0),
                (_qt_12 - 1.0, _qu_12 - 1.0, _qv_12 - 1.0),
            )
            if self._nemo_ws_test_hooks.expose_tracer_transport_stage:
                _stage_index = (
                    self._nemo_ws_test_hooks.expose_tracer_transport_stage - 1)
                if _stage_index not in (0, 1, 2):
                    raise ValueError(
                        "expose_tracer_transport_stage must be 0, 1, 2, or 3")
                _exposed_geom = _nemo_ws_live_stage_geometry[_stage_index]
                _exposed_w = (
                    _momentum_stage_w(_exposed_geom)
                    if self._nemo_ws_test_hooks.expose_stage_momentum_w
                    else _exposed_geom[2])
                _nemo_ws_exposed_tracer_transport = (
                    _exposed_geom[7],
                    _exposed_geom[8],
                    (_exposed_w
                     if self._nemo_ws_test_hooks.expose_tracer_transport_as_ww
                     else _exposed_w
                     * jnp.asarray(_grid.area_T)[..., None]),
                )
            if self._nemo_ws_test_hooks.expose_stage_face_r3:
                _r3_stage = _nemo_ws_live_stage_qco[
                    self._nemo_ws_test_hooks.expose_stage_face_r3 - 1]
                _nemo_ws_exposed_stage_face_r3 = (
                    _r3_stage[1][..., 0], _r3_stage[2][..., 0])
            u3_corr = u3_corr.at[:, -1].set(u3_corr[:, 0])
            state_new = state_new._replace(
                u=state_new.u.replace(data=u3_corr),
                v=state_new.v.replace(data=v3_corr),
            )
            if self._nemo_ws_test_hooks.expose_momentum_stage == 1:
                _nemo_ws_exposed_momentum_stage = (u1_corr, v1_corr)
            elif self._nemo_ws_test_hooks.expose_momentum_stage == 2:
                _nemo_ws_exposed_momentum_stage = (u2_corr, v2_corr)
            if _use_transport_reconcile:
                def _transport_stage(u_in, v_in):
                    mean_u = (jnp.sum(u_in * h_u_pre, axis=-1) / H_u_pre
                              * state.u_mask.data)
                    mean_v = (jnp.sum(v_in * h_v_pre, axis=-1) / H_v_pre
                              * state.v_mask.data)
                    return (
                        (u_in + (transport_target_u - mean_u)[..., None])
                        * _ws_stage_u_mask,
                        (v_in + (transport_target_v - mean_v)[..., None])
                        * _ws_stage_v_mask,
                    )
                _nemo_ws_velocity_stages = (
                    _transport_stage(u0, v0),
                    _transport_stage(u1_corr, v1_corr),
                    _transport_stage(u2_corr, v2_corr),
                )
            else:
                _nemo_ws_velocity_stages = (
                    (u0, v0), (u1_corr, v1_corr), (u2_corr, v2_corr))

        # NEMO's WZV call 2 consumes the raw boxcar pssh(Kaa) produced by
        # dyn_spg_ts (:991,1003). Keep that exact within-step operand before
        # legoESM's separate global eta-drift projection below modifies the
        # model state used by tracers and the committed next step.
        _eta_after_spg_literal = state_new.eta.data

        # 6b. Issue #271: project out global mean-eta drift right after
        # the barotropic solve, BEFORE the flux-form tracer step
        # recomputes ``h_k_new`` and consumes ``Hu_avg``.  Applying the
        # correction here keeps the tracer step's layer thicknesses
        # consistent with the corrected eta, and preserves the
        # barotropic-solver invariant ``div(Hu_avg) ==
        # (eta_old - eta_new) / dt`` up to a global mean drift that the
        # projection is exactly removing.
        #
        # Target volume = vol(eta_old) + dt * area-weighted F_slow_eta.
        # The implicit-CN solver already conserves this internally so
        # the correction is round-off; the explicit substepping path and
        # any partial-cell-induced bias get fixed here.
        #
        # All area-weighted sums are computed in ``ocean_diagnostics``
        # accumulation precision (f64 even when state runs at f32) so
        # the ``target_mass - actual_mass`` subtraction does not lose
        # the entire signal to catastrophic cancellation.
        if _cfg_b.fix_eta_drift:
            from legoesm.ocean.conservation import ocean_global_sum
            from legoesm.core.precision import cast as _cast

            _M = "ocean_diagnostics"
            mask_eta = state.land_mask.data
            area_eta = _grid.area
            # TIME LEVEL (#1226): the target volume must be the one the
            # barotropic solve actually integrated FROM.  On the MLF leap-frog
            # path the solve is SEEDED FROM Nbb (`_barotropic_before_state`
            # -> eta_init=_eta_bef above), so it returns eta(Naa) whose volume
            # is vol(eta_Nbb) -- targeting vol(eta_Nnn) instead injects a
            # uniform shift ~ mean(eta_nn) - mean(eta_bb) EVERY step.  That is a
            # state DIFFERENCE, not a tendency, so it is O(dt^0) and does not
            # shrink under timestep refinement; it breaks the very invariant
            # this block's docstring claims to preserve, and with it the
            # flux-form tracer scheme's constancy preservation (a uniform tracer
            # stops staying uniform).  Measured: disabling fix_eta_drift
            # entirely improves constancy 3.279e-05 -> 2.313e-05.
            if _barotropic_before_state is not None:
                eta_old_d = _barotropic_before_state[0]
            else:
                eta_old_d = state.eta.data
            eta_new_d = state_new.eta.data

            mask_acc = _cast(mask_eta, _M, "accumulate")
            area_acc = _cast(area_eta, _M, "accumulate")
            eta_old_acc = _cast(eta_old_d, _M, "accumulate")
            eta_new_acc = _cast(eta_new_d, _M, "accumulate")
            wa = area_acc * mask_acc

            target_local = jnp.sum(eta_old_acc * wa)
            if F_slow_eta is not None:
                F_acc = _cast(F_slow_eta, _M, "accumulate")
                target_local = target_local + dt * jnp.sum(F_acc * wa)
            actual_local = jnp.sum(eta_new_acc * wa)
            ocean_area_local = jnp.sum(wa)
            # MPI-aware reduction: returns global totals on the
            # distributed path, identity on a single rank.  Stack so
            # the reduction is one allreduce call.
            target_mass, actual_mass, ocean_area = ocean_global_sum(
                jnp.stack([target_local, actual_local, ocean_area_local])
            )
            eta_correction = (target_mass - actual_mass) / jnp.maximum(
                ocean_area, 1.0e-30
            )
            eta_fixed = eta_new_d + eta_correction.astype(eta_new_d.dtype) * mask_eta
            # Apply the same mass-conserving floor that the barotropic
            # solvers use, so a hard-floored cell does not silently
            # break the volume guarantee we just enforced.
            if _cfg_b.min_water_column_m is not None:
                from legoesm.ocean.dynamics.eta_floor import (
                    clamp_and_redistribute as _clamp_redistribute,
                )
                eta_floor = (
                    jnp.asarray(
                        _cfg_b.min_water_column_m,
                        dtype=eta_fixed.dtype,
                    )
                    - state_new.H_bathy.data
                )
                eta_fixed = _clamp_redistribute(
                    eta_fixed, eta_floor, mask_eta, area_eta,
                )
            state_new = state_new._replace(
                eta=state_new.eta.replace(data=eta_fixed),
            )

        # --- Prescribed-flow lever: PIN THE CIRCULATION here, after the
        # (discarded) momentum + barotropic solves and BEFORE the flux-form
        # tracer step (section 7), so the tracer mass fluxes, the
        # continuity-diagnosed w, and h_k_new are ALL built from the pinned
        # flow — resetting after the step would not isolate (the intra-step
        # advection would still use the post-momentum flow; codex finding 1).
        #
        # Sign/conservation convention at this splice: the flux-form update
        # below is  h_new·T_new = h_old·T_mid − dt·[div_h(mf·T) + Δ_k(ẇ·T)]
        # with mf = h·u positive along +x/+y and ẇ the z-star transport
        # velocity (zero at surface and bottom).  The lever changes ONLY the
        # advecting mf (and hence ẇ, diagnosed from div(mf)); every T/S
        # source/sink — physics tendencies, surface forcing, penetrating SW,
        # virtual salt, restoring, implicit vertical mixing — is untouched,
        # so T/S column budgets differ from the free run only through the
        # (zeroed/held) advection, and the global ∫h·T budget closes by the
        # divergence theorem exactly as in the free run.
        #   "zero"   → u = v = eta = 0: mf = 0 ⇒ ẇ = 0 (pure column physics).
        #              NB: start from an eta≡0 state (rest/WOA cold start) —
        #              a nonzero entry eta incurs a one-time h_old/h_new
        #              tracer rescale on the first step (eta jumps to 0).
        #   "frozen" → u/v/eta pinned to the step-ENTRY values: tracers are
        #              advected by the held flow; h_new == h_old exactly.  A
        #              divergent held flow pumps tracer between columns (its
        #              physical advective convergence) while the global
        #              budget still closes.
        if _pflow is not None:
            if _pflow == "zero":
                _u_pin = jnp.zeros_like(state_new.u.data)
                _v_pin = jnp.zeros_like(state_new.v.data)
                _eta_pin = jnp.zeros_like(state_new.eta.data)
            else:  # "frozen" — the only other value (validated at construction)
                _u_pin = state.u.data
                _v_pin = state.v.data
                _eta_pin = state.eta.data
            state_new = state_new._replace(
                u=state_new.u.replace(data=_u_pin),
                v=state_new.v.replace(data=_v_pin),
                eta=state_new.eta.replace(data=_eta_pin),
            )

        # 7. Flux-form tracer update using full 3D velocity
        #
        # The barotropic solver returns Hu_avg (time-averaged depth-
        # integrated transport) consistent with the continuity equation
        # that produced h_new.  For tracer advection we need per-layer
        # mass fluxes that:
        #   (a) preserve the baroclinic velocity shear (needed for
        #       Ekman pumping and vertical tracer transport), and
        #   (b) have depth-integrated transport matching Hu_avg (needed
        #       for consistency with the barotropic continuity).
        #
        # We take the full 3D velocity from state_new (which preserves
        # baroclinic structure) and apply a uniform barotropic correction
        # so that sum_k(h_k * u_corrected_k) = Hu_avg exactly.
        # (Hallberg & Adcroft 2009, Shchepetkin & McWilliams 2005).
        _min_uface_op = min_cell_to_uface
        _min_vface_op = lambda f: min_cell_to_vface(f, _grid)
        mask = state.land_mask.data

        # Layer thickness at face points (min-rule, partial-cell aware
        # and consistent with the barotropic solver and slow forcing).
        h_u_old = _min_uface_op(h_k_old)        # (n_lat, n_lon+1, nlev)
        h_v_old = _min_vface_op(h_k_old)        # (n_lat+1, n_lon, nlev)
        H_u_old = jnp.sum(h_u_old, axis=-1)     # (n_lat, n_lon+1)
        H_v_old = jnp.sum(h_v_old, axis=-1)     # (n_lat+1, n_lon)

        # 3D face masks for the tracer mass flux.  For partial cells the
        # 2D u_mask/v_mask are non-zero at the topographic-step face
        # (both surface columns are wet) but the face must be closed
        # below the shallower seafloor.  Using compute_face_masks_3d on
        # the partial coord's is_active gives the correct per-level
        # closed-wall faces.  For pure z\\* the 3D mask collapses to the
        # 2D mask broadcast across all levels — bit-exact backwards-compat.
        # Likewise, ``active_3d`` gates inactive cells (below the
        # partial seafloor) where h_k_old = h_k_new = 0; without this
        # gate, the floor in ``tr_new = hT_new / max(h_k_new, 1e-10)``
        # amplifies tiny float-precision residuals into huge spurious
        # tracer values inside the ground.
        if isinstance(_zc, OceanPartialCellCoordinate):
            u_mask_3d_tracer, v_mask_3d_tracer = compute_face_masks_3d(
                _zc.is_active, _grid,
            )
            u_mask_3d_tracer = u_mask_3d_tracer.astype(h_u_old.dtype)
            v_mask_3d_tracer = v_mask_3d_tracer.astype(h_v_old.dtype)
            active_3d = _zc.is_active.astype(h_u_old.dtype)
        else:
            u_mask_3d_tracer = state.u_mask.data[..., jnp.newaxis]
            v_mask_3d_tracer = state.v_mask.data[..., jnp.newaxis]
            active_3d = mask_3d

        # Full 3D velocity (barotropic + baroclinic) from state after
        # barotropic correction.  The generic path below preserves the
        # baroclinic perturbation u' = u - U_bar and replaces the barotropic
        # component with the time-averaged U_bar_avg.  DINO's literal QCO
        # path instead executes dynspg_ts.F90:1170-1174 on Kmm itself before
        # tra_adv points zptu/zptv at that state (traadv.F90:301-304).
        u_3d = state_new.u.data   # (n_lat, n_lon+1, nlev)
        v_3d = state_new.v.data   # (n_lat+1, n_lon, nlev)

        # Correct the barotropic component so that depth-integrated
        # transport matches Hu_avg exactly.  The correction is the
        # difference between <H*U> (time-averaged transport) and
        # <U>*H (time-averaged velocity times pre-barotropic H).
        if _pflow is not None:
            # Prescribed flow: NO barotropic transport correction.  u_3d/v_3d
            # already carry the pinned flow (splice above); Hu_avg/Hv_avg from
            # the DISCARDED barotropic solve must not leak into the tracer
            # mass fluxes.  mass_flux = h·u_pin exactly — for "zero" the
            # fluxes vanish and the diagnosed w below is identically zero.
            u_corrected = u_3d
            v_corrected = v_3d
        elif getattr(_cfg_b, "wzv_call2_evaluation", "generic") == "nemo_literal":
            # One coupled Kmm operation feeds both tracer horizontal fluxes
            # and WZV call 2.  Reusing the literal execute/undo primitive here
            # prevents the historical half-state where W saw the corrected
            # velocity but tra_adv still saw the generic state_new velocity.
            u_corrected, v_corrected, _, _ = nemo_qco_kmm_velocity_cycle(
                state.eta.data, state.u.data, state.v.data,
                Hu_avg, Hv_avg, _zc, u_mask_3d_tracer, v_mask_3d_tracer,
                _grid)
            # traadv.F90:328-331 consumes the SAME live Kmm QCO face
            # thickness as the literal velocity cycle, not lego's generic
            # min-of-neighbour thickness. Build the native east/north faces
            # from the raw bridge operands, then map once to the redundant
            # west/south layout used by the tracer core.
            _e3t0 = jnp.asarray(
                _zc.nemo_e3t_0, dtype=state.eta.data.dtype)[
                    ..., :u_corrected.shape[-1]]
            _raw_umask = jnp.asarray(
                u_mask_3d_tracer[:, 1:, :], dtype=state.eta.data.dtype)
            _raw_vmask = jnp.asarray(
                v_mask_3d_tracer[1:, :, :], dtype=state.eta.data.dtype)
            # The SAME shared builder the WS-RK3 stage transport calls
            # (_nemo_ws_qco_stage_faces): NEMO's MLF dom_qco_r3c
            # (domqco.F90:166-169) and dom_qco_r3c_RK3 (:219-222) are the
            # same r3u statement, so legoESM has one implementation reached
            # by both lanes.  Only the OPERAND SOURCE differs -- this lane's
            # cards carry NEMO's own hu_0/e1e2* on z_coord.nemo_*.
            _h_u_tracer, _h_v_tracer, _, _ = nemo_qco_live_face_geometry_cgrid(
                state.eta.data, _e3t0, _e3t0, _raw_umask, _raw_vmask,
                *nemo_qco_mesh_operands(_zc, state.eta.data.dtype))
        else:
            # H + Hu reductions per face share the h_u_old/h_v_old weight
            # on the level axis — fuse into one stack each.  Keep this entire
            # legacy arm textually isolated so non-fidelity cards retain their
            # prior arithmetic topology byte-for-byte.
            _u_pair = jnp.sum(
                jnp.stack([h_u_old, u_3d * h_u_old], axis=-1), axis=-2)
            H_u_old, Hu_3d = _u_pair[..., 0], _u_pair[..., 1]
            _v_pair = jnp.sum(
                jnp.stack([h_v_old, v_3d * h_v_old], axis=-1), axis=-2)
            H_v_old, Hv_3d = _v_pair[..., 0], _v_pair[..., 1]
            delta_U = (Hu_avg - Hu_3d) / jnp.maximum(H_u_old, 1e-10)
            delta_V = (Hv_avg - Hv_3d) / jnp.maximum(H_v_old, 1e-10)
            u_corrected = u_3d + delta_U[..., jnp.newaxis]
            v_corrected = v_3d + delta_V[..., jnp.newaxis]

        # Per-layer mass fluxes with full 3D velocity structure.
        # Unlike the previous barotropic-only distribution (which gave
        # uniform velocity at all depths and identically zero w),
        # this preserves baroclinic shear and produces non-zero vertical
        # velocity from Ekman pumping/suction.
        if getattr(_cfg_b, "wzv_call2_evaluation", "generic") == "nemo_literal":
            mass_flux_u = _h_u_tracer * u_corrected * u_mask_3d_tracer
            mass_flux_v = _h_v_tracer * v_corrected * v_mask_3d_tracer
        else:
            # Preserve the generic/off statements byte-for-byte.
            mass_flux_u = h_u_old * u_corrected * u_mask_3d_tracer
            mass_flux_v = h_v_old * v_corrected * v_mask_3d_tracer

        # Flux-form tracer update (horizontal + vertical)
        #
        # Both horizontal and vertical transport use the barotropic-averaged
        # per-layer divergence for consistency:
        #   h_new * T_new = h_old * T_mid
        #     - dt * div_h(mf_k * T_face_h)        [horizontal flux]
        #     - dt * (w_{k-1/2}*T_{k-1/2} - w_{k+1/2}*T_{k+1/2})  [vertical flux]
        #
        # The horizontal flux integrates to zero by the 2D divergence theorem.
        # The vertical flux telescopes to surface/bottom (both zero).
        # Total conservation is exact.


        h_k_new = compute_layer_thickness(
            state_new.eta.data, state_new.H_bathy.data, _zc,
            min_water_column_m=_cfg_b.min_water_column_m,
        )

        # Diagnose w from barotropic-averaged per-layer divergence
        # (consistent with the horizontal transport used for tracers).
        # mass_flux_u/v are thickness-weighted (h*u), so flux_div_k
        # is div(h*u) [m/s] and already includes layer thickness.
        flux_div_k = divergence_cgrid(mass_flux_u, mass_flux_v, _grid)
        # NOTE (#1226, verified numerically 2026-07-26): the sigma-correction
        # form below and NEMO's literal continuity form (sshwzv.F90:198-206,
        # which folds the ACTUAL per-layer thickness tendency
        # (h_new - h_old)/dt into the integrand) give IDENTICAL results here to
        # 4 significant figures on the content-conservation probe.  That means
        # legoESM's split-explicit barotropic eta IS already consistent with the
        # column-integrated div(h*u), so the inferred deta_dt = w_euler[...,0]
        # equals the actual increment.  A `dh_dt=` path was implemented, tested,
        # found inert, and removed rather than carried as dead weight.
        if getattr(_cfg_b, "wzv_call2_evaluation", "generic") == "nemo_literal":
            # NEMO call 2 (stpmlf.F90:350-412; sshwzv.F90:218-227): the
            # dyn_spg-corrected Kmm velocity feeds a SECOND div_hor and that
            # divergence is paired with the ACTUAL barotropic Kaa r3t.  Round
            # 42 measured either operand alone as ~48-50x worse in squared
            # call-delta energy, so this is one coupled operation.
            _eta_before_field = getattr(state, "eta_before", None)
            _eta_before_wzv = (state.eta.data if _eta_before_field is None
                               else _eta_before_field.data)
            w_baro, _, _ = nemo_qco_wzv_operands(
                state.eta.data, _eta_before_wzv, state.u.data, state.v.data,
                _grid, _zc, u_mask_3d_tracer, v_mask_3d_tracer, active_3d,
                dt, eta_after_override=_eta_after_spg_literal,
                transport_after_override=(Hu_avg, Hv_avg))
            w_baro = jax.lax.optimization_barrier(w_baro)
        else:
            w_baro = diagnose_w_from_flux_div(
                flux_div_k, _zc, thickness_weighted=True,
            )

        _nemo_ws_stage_transport_geometry = None
        _nemo_ws_aimp_tracer_w = None
        _nemo_ws_aimp_momentum_w_u = None
        _nemo_ws_aimp_momentum_w_v = None
        if (getattr(_cfg_b, "tracer_time_integrator", "euler") == "rk3_ws"
                and self._nemo_ws_test_hooks.kmm_tracer_transports):
            if _nemo_ws_velocity_stages is None:
                raise ValueError(
                    "nemo_kmm tracer transports require materialized WS "
                    "momentum stages")
            if _nemo_ws_live_stage_geometry is not None:
                # stprk3_stg.F90:257-304: the momentum program's Kmm triplets
                # are the tracer transports too (tra_adv_trp reuses zF).
                _nemo_ws_stage_transport_geometry = (
                    _nemo_ws_live_stage_geometry)
            else:
                # UNREACHABLE since the S-30/S-12 collapse, kept only because
                # deleting it is a separate change with its own gate: with one
                # WS ladder, `_nemo_ws_live_stage_geometry` is set for every
                # rk3_ws momentum step, and `_validate_config` requires rk3_ws
                # for BOTH integrators or for neither, so a tracer program on
                # rk3_ws always finds it.  (Historically this rebuilt the Kmm
                # ladder for the arm that skipped the whole post-solve ladder.)
                # Stage 1 reads Kbb,
                # stage 2 the stage-1 Kaa (1/3), stage 3 the stage-2 Kaa
                # (1/2): stprk3_stg.F90:160-168,195-213,225-234 before
                # :250-303 builds zFu/zFv/zFw from Kmm.
                h_k_one_third = h_k_old + (h_k_new - h_k_old) / 3.0
                h_k_one_half = 0.5 * (h_k_old + h_k_new)
                stage_h_k = (h_k_old, h_k_one_third, h_k_one_half)
                _eta_bb = state.eta.data
                _eta_aa = state_new.eta.data
                stage_eta = (
                    _eta_bb,
                    _eta_bb + (_eta_aa - _eta_bb) / 3.0,
                    0.5 * (_eta_bb + _eta_aa),
                )
                _h_ref_legacy = compute_layer_thickness(
                    jnp.zeros_like(_eta_bb), state.H_bathy.data, _zc,
                    min_water_column_m=_cfg_b.min_water_column_m)
                _nemo_ws_stage_transport_geometry = tuple(
                    _nemo_ws_stage_transport(
                        velocity, h_stage, stage_index,
                        eta_stage=eta_stage, h_ref=_h_ref_legacy,
                        Hu_avg=Hu_avg, Hv_avg=Hv_avg,
                        u_mask_3d=u_mask_3d_tracer,
                        v_mask_3d=v_mask_3d_tracer, grid=_grid,
                        z_coord=_zc, H_bathy=state.H_bathy.data,
                        config=_cfg_b, dt=dt,
                        legacy_min_face_thickness=(
                            self._nemo_ws_test_hooks
                            .legacy_stage_min_face_thickness),
                        legacy_aimp_midpoint_w_metric=(
                            self._nemo_ws_test_hooks
                            .legacy_aimp_midpoint_w_metric))
                    for stage_index, (velocity, h_stage, eta_stage)
                    in enumerate(zip(
                        _nemo_ws_velocity_stages, stage_h_k, stage_eta,
                        strict=True)))

            if (getattr(_cfg_b, "adaptive_implicit_vertadv", False)
                    and not self._nemo_ws_test_hooks.disable_adaptive_implicit_momentum):
                # stprk3_stg.F90:298-302 partitions the stage-3 ww.  The
                # explicit share already entered the stage-3 momentum RHS
                # (dyn_adv, :331-334) and the tracer FCT above; dynzdf.F90:
                # 252-270 folds the implicit share into the one stage-3
                # matrix, area-weighted to the velocity faces.
                _nemo_ws_aimp_tracer_w = _nemo_ws_stage_transport_geometry[2][6]
                _area_wi = (_grid.area_T[..., None]
                            * _nemo_ws_aimp_tracer_w)
                _area_u = (_grid.dx_u * _grid.dy_u)[..., None]
                _area_v = (_grid.dx_v * _grid.dy_v)[..., None]
                _nemo_ws_aimp_momentum_w_u = (
                    interp_cell_to_uface(_area_wi)
                    / jnp.maximum(_area_u, 1.0e-30))
                _nemo_ws_aimp_momentum_w_v = (
                    interp_cell_to_vface(_area_wi, _grid)
                    / jnp.maximum(_area_v, 1.0e-30))

        # Advecting mass fluxes for the TRACER scheme.  Default = the base
        # (momentum/continuity) mass flux; when GM runs with
        # gm_bolus_advection="through_fct" the eddy-induced (bolus) transport is
        # added to THESE (only) below, so the bolus passes through the monotone
        # FCT limiter (NEMO traadv) without touching the momentum/continuity/eta
        # mass fluxes (which keep using mass_flux_u/v and w_baro).
        mass_flux_u_tr, mass_flux_v_tr, w_baro_tr = (
            mass_flux_u, mass_flux_v, w_baro)

        # 7b. Adaptive-implicit vertical momentum advection
        #     (Shchepetkin 2015 / NEMO ``ln_zad_Aimp``).  The explicit
        #     in-tendency vertical momentum advection has no vertical-CFL
        #     limit and amplifies a spurious ``w`` super-exponentially in
        #     thin cells (the OMIP cold-start "vertadv" runaway).  When
        #     ``config.adaptive_implicit_vertadv`` is set, the PE tendency
        #     skips that explicit term and it is applied here instead,
        #     after the barotropic solve, on the barotropic-consistent
        #     ``w_baro`` (the same vertical velocity that advects tracers).
        #     It acts on the baroclinic perturbation ``u' = u - U_bar``
        #     (depth-mean removed — the barotropic mode is owned by the
        #     barotropic solver), exactly like the explicit scheme, then
        #     restores ``U_bar``.  Unconditionally stable + conservative.
        #     No-op (and bit-exact) when the flag is off.
        #     SKIPPED under the prescribed-flow lever: this block is
        #     momentum-ONLY (it rewrites state_new.u/v, which the lever pins),
        #     so running it would unpin the prescribed flow mid-step for zero
        #     physical effect on T/S.
        if (getattr(_cfg_b, "adaptive_implicit_vertadv", False)
                and not self._nemo_ws_test_hooks.disable_adaptive_implicit_momentum
                and getattr(_cfg_b, "momentum_time_integrator", "euler")
                != "rk3_ws"
                and _pflow is None):
            from legoesm.ocean.vertical import (
                adaptive_implicit_vertical_momentum_advection,
            )
            # w_baro at cell centers -> momentum faces (fold-aware for v).
            w_u_half = interp_cell_to_uface(w_baro)            # (lat, lon+1, nlev+1)
            w_v_half = interp_cell_to_vface(w_baro, _grid)  # (lat+1, lon, nlev+1)
            # Depth-mean (barotropic) velocity at the faces, from the
            # already-computed thickness-weighted transports (lines above).
            U_bar = (Hu_3d / jnp.maximum(H_u_old, 1e-10))[..., jnp.newaxis]
            V_bar = (Hv_3d / jnp.maximum(H_v_old, 1e-10))[..., jnp.newaxis]
            u_face_active = jnp.broadcast_to(u_mask_3d_tracer, u_3d.shape)
            v_face_active = jnp.broadcast_to(v_mask_3d_tracer, v_3d.shape)
            _vertical_scheme = getattr(
                _cfg_b, "vertical_momentum_scheme", "upwind_perturbation")
            if _vertical_scheme == "nemo_up3":
                # NEMO dynadv_up3 advects full uu/vv.  Its adaptive split
                # partitions only the vertical transport (stprk3_stg.F90:
                # 284-300), so use the same UP3 explicit flux on w_exp and
                # the canonical implicit upwind solve on w_imp.
                u_adv = adaptive_implicit_vertical_momentum_advection(
                    u_3d, w_u_half, h_u_old, dt,
                    face_active=u_face_active, explicit_scheme="nemo_up3")
                v_adv = adaptive_implicit_vertical_momentum_advection(
                    v_3d, w_v_half, h_v_old, dt,
                    face_active=v_face_active, explicit_scheme="nemo_up3")
            else:
                u_adv = adaptive_implicit_vertical_momentum_advection(
                    u_3d - U_bar, w_u_half, h_u_old, dt,
                    face_active=u_face_active,
                ) + U_bar
                v_adv = adaptive_implicit_vertical_momentum_advection(
                    v_3d - V_bar, w_v_half, h_v_old, dt,
                    face_active=v_face_active,
                ) + V_bar
            # Re-apply the 2D wet mask + periodic wrap column (matches the
            # tendency path's post-update masking at u[:, -1] = u[:, 0]).
            u_adv = u_adv * u_mask_3d
            u_adv = u_adv.at[:, -1].set(u_adv[:, 0])
            v_adv = v_adv * v_mask_3d
            state_new = state_new._replace(
                u=state_new.u.replace(data=u_adv),
                v=state_new.v.replace(data=v_adv),
            )

        T_mid = state_new.T.data  # tracer after diffusion+physics Euler step
        S_mid = state_new.S.data
        # ``_ldf_state`` (nemo_mlf P1): GM/Redi's isoneutral-Redi tendency is
        # row 28's lego home (``tra_ldf -> traldf_iso_lap``, ``stpmlf.F90:437
        # pts(:,:,:,:,Kbb)``) -- so under the single-pass transcription it
        # reads the RAW Nbb tracers directly (MORE faithful than the two-pass
        # mechanism's "T_mid ~= Nbb", which was Nbb plus that discarded pass's
        # own small Euler correction).
        #
        # CORRECTED 2026-09-02 (M-01 mechanism review).  This comment used to
        # claim "GM bolus transport / K_33 / EKE production are UNCHANGED --
        # only the isoneutral tendency's (T, S) READ swaps".  That is FALSE,
        # and the code ~230 lines below says so itself: ``_T_gm_in``/
        # ``_S_gm_in`` are the tracer source for BOTH
        # ``gm_redi_density_and_jacobian`` (which builds the neutral slopes and
        # K33) AND ``gm_redi_tracer_tendency_latlon`` (which builds the bolus
        # streamfunction), and they MUST agree on their time level or K33 would
        # disagree with the flux it augments.  So the swap moves the whole
        # GM/Redi operand set, slopes included -- which is why the measured
        # tracer difference between the two stp_MLF mechanisms runs through the
        # slope denominator (d_z rho) rather than linearly in the tracer
        # perturbation.  ``eta``/``H_bathy``/``kappa`` DO stay on the step's own
        # (Nnn) state.  ``None``
        # (every other caller) ⇒ bit-identical.
        _T_gm_in = T_mid if _ldf_state is None else _ldf_state[0]
        _S_gm_in = S_mid if _ldf_state is None else _ldf_state[1]
        # The positional ``eta`` these three GM/Redi calls receive.  Naa (the
        # step's own after-ssh) on every historical lane.
        _eta_gm_in = state_new.eta.data

        # NEMO's WS-RK3 STAGE PROGRAM COMPUTES THE NEUTRAL SLOPES ONCE PER
        # STEP, ON THE BEFORE STATE, OUTSIDE THE STAGE LOOP:
        #
        #     CALL eos ( ts, Nbb, rhd )                   stprk3.F90:173
        #     CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )   stprk3.F90:174
        #     ...
        #     CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )   :195
        #     CALL stp_RK3_stg( 2, ... )  :200      CALL stp_RK3_stg( 3, ... )  :207
        #
        # ``rhd`` is the density of the BEFORE tracers, ``rn2b`` the BEFORE
        # Brunt-Vaisala frequency, and ``Kbb = Kmm = Nbb`` inside ``ldf_slp``
        # so every geometry it reads -- gdept, gdepw, e3u, e3v, e3w and the ssh
        # behind them -- is at the before level too (ldfslp.F90:143,161,
        # 212-215,226-231,290).  ``traldf_iso_a33`` then squares those
        # once-per-step slopes into ``ah_wslp2`` at each stage
        # (traldf_iso.F90:135,296-297) and ``trazdf.F90:173`` adds it to
        # ``avt``.  NOTHING in ``ah_wslp2`` carries a stage time level at all:
        # ``ahtu``/``ahtv`` have no time index (traldf_iso.F90:291-294) and the
        # only ``Kmm`` in that routine is ``akz``'s ``e3w`` (:325,:330), which
        # this card does not reach (``ln_traldf_msc = F``).  An earlier draft
        # of this comment said the ahtu average carried the stage's Kmm; an
        # independent claim review corrected it.
        #
        # legoESM built them from ``T_mid``/``S_mid`` -- the forward-Euler
        # PREDICTOR of the whole baroclinic tendency, built above -- and from
        # the after-ssh.  MEASURED on GYRE's kt=1 step: that predictor is
        # 0.3372313792464041 K away from step entry on 10199 of 21120 cells,
        # where NEMO's own stage-minus-before tracer anomaly is 7.25e-6 K, and
        # the resulting K33 reaches 9.6620886711883531e-13 m2/s where NEMO's
        # ah_wslp2 on the same step is IDENTICALLY 0.0.  Substituting one
        # component at a time into the model's own call: tracers := step entry
        # gives 1.3071545538245700e-14, eta := step entry gives
        # 9.6625031189531020e-13, BOTH give EXACTLY 0.0 on all 20416 faces.
        #
        # So on the WS-RK3 tracer lane the slope operand set IS the step-entry
        # (Kbb) state.  ``_ldf_state`` is the same statement for the
        # modified-leapfrog lane and takes precedence where it is supplied;
        # every other integrator has no stage loop for this statement to be
        # about and is untouched, bit for bit.
        if (_ldf_state is None
                and getattr(_cfg_b, "tracer_time_integrator", "euler")
                == "rk3_ws"):
            _T_gm_in = state.T.data
            _S_gm_in = state.S.data
            _eta_gm_in = state.eta.data

        # GM/Redi isopycnal mixing (if configured)
        k33_implicit = None  # vertical isoneutral diffusivity K_33 for the
        #                      implicit tracer solve (set below iff implicit_K33).
        if _cfg_b.gm_redi is not None:
            gm_cfg = _cfg_b.gm_redi
            kappa_gm_override = None
            kappa_redi_override, kappa_redi_v_override = static_kappa_redi_override(
                gm_cfg, _grid)
            eke_new = None
            eke_diss_new = None
            # Prognostic-EKE GM closure (Eden-Greatbatch): kappa_GM = c_k·L·√E
            # from the evolving eddy-energy field, and integrate E one step
            # (transport by the depth-mean flow + semi-implicit source/sink).
            # Gated on gm_cfg.eke -> default None leaves the existing path
            # (constant / Visbeck) bit-identical.
            if gm_cfg.eke is not None:
                eke_cfg = gm_cfg.eke
                lm = state.land_mask.data
                # --- Hallberg resolution-function EKE-budget coupling (codex
                # MED-3 r2). The tracer path applies kappa_eff = f_res*kappa
                # (inside gm_redi_tracer_tendency_latlon); the SAME f_res must
                # scale the GM-DERIVED eddy-energy production so the E budget
                # receives exactly the APE->EKE conversion the APPLIED
                # coefficient performs (MOM6 MEKE precedent). An unscaled
                # production would over-energise E — and hence the prognostic
                # kappa = c_k*L*sqrt(E) — relative to the realized GM work.
                # Redi-side terms (kappa_redi_override, -P_diss_iso, GEOMETRIC
                # kappa_n) and the barotropic B_T (kappa_u) stay UNSCALED (the
                # taper is GM-only). Same inputs as the tendency's scaling site
                # (broadcast grid.f, sqrt(cell area)) => bit-identical factor.
                # Static Python gate; None (default off) => byte-identical.
                _resfn_scale = None
                if getattr(gm_cfg, "resolution_function", False):
                    _resfn_scale = gm_resolution_factor(
                        jnp.broadcast_to(_grid.f, lm.shape),
                        jnp.sqrt(_grid.area),
                        gm_cfg.resfn_gamma, gm_cfg.resfn_cbcl_ms,
                    )
                if eke_cfg.eke_3d:
                    # 3-D (depth-resolved) prognostic-EKE path: E lives on the
                    # interior interfaces (W-grid, nlev-1), the GM/Redi override
                    # kappa is a 3-D interface field, and E evolves by the
                    # depth-resolved source/sink + implicit vertical EKE diffusion
                    # + per-interface horizontal transport (Veros's 3-D vs.eke).
                    eke_new, kappa_gm_override, eke_diss_new = self._eke_3d_step(
                        state, state_new, T_mid, S_mid, gm_cfg, eke_cfg, lm,
                        tend.A_v, dt,
                        Ah_visc_u=tend.Ah_visc_u, Ah_visc_v=tend.Ah_visc_v,
                        Ah_kediss_cell=tend.Ah_kediss_cell,
                        grid=_grid,
                        resfn_scale=_resfn_scale,
                    z_coord=z_coord, config=config)
                elif eke_cfg.closure == "geometric":
                    # GEOMETRIC closure (Torres et al. 2025, JAMES,
                    # doi:10.1029/2025MS005394): depth-INTEGRATED 2-D EKE
                    # budget (Eq. 1) — ``state.eke`` carries ∫EKE dz [m³/s²].
                    # Static Python dispatch on the config literal (validated
                    # at construction); the default closure is bit-identical.
                    geom = eke_cfg.geometric
                    E_in = state.eke.data if state.eke is not None else None
                    (E, kappa_gm_override, kappa_n_geom, prod_bc, L_eff,
                     dz_geom) = compute_geometric_step_kappa(
                        T_mid, S_mid, state_new.eta.data, state_new.H_bathy.data,
                        E_in, _grid, _zc, gm_cfg,
                        eos=_cfg_b.eos, eos_linear=_cfg_b.eos_linear,
                        mask=lm,
                        rho_0=_cfg_b.constants.rho_0,
                        g=_cfg_b.constants.g,
                        omega=_cfg_b.constants.Omega,
                        r_earth=_cfg_b.constants.R_earth,
                    )
                    # Barotropic production B_T = ∫κ_u|∇h u_h|² dz (Eq. 3)
                    # from the start-of-step velocities (the same time level
                    # that advects E below).
                    prod_bt = geometric_barotropic_production(
                        state.u.data, state.v.data, _grid, geom.kappa_u,
                        dz_geom, lm, state.u_mask.data, state.v_mask.data,
                        z_coord=_zc,
                    )
                    # Shim config routing the GEOMETRIC coefficients through
                    # the SHARED 2-D transport + semi-implicit fold:
                    # k_iso←kappa_e (T_e diffusion, Eq. 5) and
                    # c_eps←c_eps_geometric, which with L=L_eff=R_d·√H makes
                    # the fold rate c_eps·√I/L_eff = C_ε·√(I/H)/R_d — the
                    # EXACT Eq.-4 dissipation linearised implicitly (see
                    # geometric_dissipation_length).  No duplicate numerics.
                    shim_cfg = eke_cfg._replace(
                        c_eps=geom.c_eps_geometric, k_iso=geom.kappa_e)
                    # Advecting flow: ∇h·(u_h ∫E dz) with the depth-mean flow
                    # (paper Eq. 1 / Appendix D "integrated transport" with
                    # φ(z)=1) — the same level-mean U_bar construction as the
                    # EG 2-D path (conserves the area integral of E).
                    U_bar = jnp.mean(state.u.data, axis=-1) * state.u_mask.data
                    V_bar = jnp.mean(state.v.data, axis=-1) * state.v_mask.data
                    E_t = E + dt * eke_horizontal_transport(
                        E, U_bar, V_bar, _grid, shim_cfg,
                        lm, state.u_mask.data, state.v_mask.data,
                    )
                    # B_C + B_T explicit (both ≥ 0), D_e implicit — E ≥ 0 by
                    # construction (the paper instead zeroes D_e where E < 0,
                    # p. 5; the fold is strictly stronger — documented).
                    # Resolution-function coupling: B_C = kappa_gm·∫M⁴/N² dz
                    # is linear in the GM coefficient, so scale it by the SAME
                    # f_res the tracer flux applies to kappa_gm. B_T (kappa_u,
                    # a separate un-tapered coefficient) is NOT scaled.
                    _prod_bc_eff = (prod_bc if _resfn_scale is None
                                    else prod_bc * _resfn_scale)
                    E_new = eke_apply_local_source(
                        E_t, jnp.zeros_like(E_t), L_eff, shim_cfg, dt,
                        production_override=_prod_bc_eff + prod_bt,
                    )
                    eke_new = Field(data=E_new * lm, name="eke",
                                    dims=("lat", "lon"), units="m^3/s^2")
                    # κ_n (Eq. 7) → Redi tracer diffusivity (EKE-GM+N).  A
                    # genuinely T-point field (unlike the static lat-scaling
                    # case) -> clear any stale v-face override so it is not
                    # silently paired with the wrong T-value.
                    if geom.kappa_n_coupling:
                        kappa_redi_override = kappa_n_geom
                        kappa_redi_v_override = None
                else:
                    if state.eke is not None:
                        E = state.eke.data
                    else:
                        E = jnp.full(lm.shape, eke_cfg.e_min, dtype=T_mid.dtype)
                    kappa_gm_override, sigma_bar, L = compute_eke_step_kappa(
                        T_mid, S_mid, state_new.eta.data, state_new.H_bathy.data,
                        E, _grid, _zc, gm_cfg,
                        eos=_cfg_b.eos, eos_linear=_cfg_b.eos_linear,
                        mask=lm,
                        rho_0=_cfg_b.constants.rho_0,
                        g=_cfg_b.constants.g,
                        omega=_cfg_b.constants.Omega,
                        r_earth=_cfg_b.constants.R_earth,
                    )
                    # Depth-mean advecting flow (level-mean; preserves the periodic
                    # wrap so the transport conserves the area-integral of E).
                    U_bar = jnp.mean(state.u.data, axis=-1) * state.u_mask.data
                    V_bar = jnp.mean(state.v.data, axis=-1) * state.v_mask.data
                    E_t = E + dt * eke_horizontal_transport(
                        E, U_bar, V_bar, _grid, eke_cfg,
                        lm, state.u_mask.data, state.v_mask.data,
                    )
                    # Resolution-function coupling: scale the parameterized
                    # production kappa_GM·sigma² by the SAME f_res the tracer
                    # flux applies to kappa_GM (None => bit-identical).
                    E_new = eke_apply_local_source(E_t, sigma_bar, L, eke_cfg, dt,
                                                   production_scale=_resfn_scale)
                    eke_new = Field(data=E_new * lm, name="eke",
                                    dims=("lat", "lon"), units="m^2/s^2")
                # K_iso = K_gm (Veros enable_eke_isopycnal_diffusion): drive the
                # Redi tracer diffusivity from the same prognostic kappa as GM.
                # kappa_gm_override is a T-point field -> clear any stale
                # static-override v-face field (same reasoning as kappa_n_geom).
                if eke_cfg.isopycnal_diffusion:
                    kappa_redi_override = kappa_gm_override
                    kappa_redi_v_override = None
            # Hoist the shared in-situ density (2-iteration EOS coupling) +
            # z* Jacobian: when implicit_K33 the tracer tendency AND the K_33
            # diagonal both need EXACTLY these from the same
            # (T_mid,S_mid,eta,H_bathy), so compute once and thread into both
            # (scaling review lever #3).  None when not implicit_K33 ⇒ the
            # tracer tendency computes them inline, bit-identical to before.
            _gm_dens_jac = None
            _gm_native_prd_J = None
            _gm_native_prd_TS = None
            _gm_native_pn2 = None
            _gm_native_e3w = None
            if getattr(gm_cfg, "slope_prd_geometry_stage", "current_step") == "before_step":
                # NEMO's Kbb operand maps to the shifting BEFORE fields only
                # in the MLF integrator.  Forward Euler never advances those
                # carry fields, so its step-entry current state is the Kbb-
                # equivalent; reading a bridged eta_before/T_before there
                # would freeze prd at the restart forever.
                _use_mlf_before = (
                    getattr(_cfg_b, "outer_integrator", "forward_euler")
                    in ("leapfrog", "nemo_mlf"))
                _eta_slope = (
                    state.eta_before.data
                    if _use_mlf_before and state.eta_before is not None
                    else state.eta.data)
                from legoesm.ocean.eos import nemo_r3t_stretch
                _gm_native_prd_J = nemo_r3t_stretch(
                    _zc, _eta_slope, state_new.H_bathy.data)
                _gm_native_prd_TS = (
                    state.T_before.data
                    if _use_mlf_before and state.T_before is not None
                    else state.T.data,
                    state.S_before.data
                    if _use_mlf_before and state.S_before is not None
                    else state.S.data,
                )
            elif getattr(gm_cfg, "slope_prd_geometry_stage", "current_step") != "current_step":
                raise ValueError(
                    "GMRediConfig.slope_prd_geometry_stage must be 'current_step' "
                    f"or 'before_step', got {gm_cfg.slope_prd_geometry_stage!r}")
            _slope_n2_eval = getattr(
                gm_cfg, "slope_n2_evaluation", "recompute")
            if _slope_n2_eval == "carried_step_entry":
                if _tke_n2_bundle is None:
                    raise ValueError(
                        "GMRediConfig.slope_n2_evaluation='carried_step_entry' "
                        "requires the pre-zdf_phy TKE N2 bundle")
                _gm_native_pn2 = _tke_n2_bundle.rn2b
                _gm_native_e3w = _tke_n2_bundle.e3w_Kmm
                _gm_surface_e3w = getattr(
                    _tke_n2_bundle, "e3w_surface_Kmm", None)
                if (_gm_surface_e3w is not None
                        and _gm_native_e3w.shape[-1]
                        == state.T.data.shape[-1] - 1):
                    _gm_native_e3w = jnp.concatenate(
                        [_gm_surface_e3w, _gm_native_e3w], axis=-1)
            elif _slope_n2_eval != "recompute":
                raise ValueError(
                    "GMRediConfig.slope_n2_evaluation must be 'recompute' or "
                    f"'carried_step_entry', got {_slope_n2_eval!r}")
            if gm_cfg.implicit_K33:
                # ``_T_gm_in``/``_S_gm_in`` (Nbb under nemo_mlf, else T_mid):
                # this hoisted density/jacobian is reused by BOTH the K33
                # diagonal AND the tendency call below, so it MUST be built
                # from the SAME tracer source the tendency call reads -- else
                # K33 and the isoneutral flux it augments would disagree on
                # which time level they came from.
                _gm_dens_jac = gm_redi_density_and_jacobian(
                    _T_gm_in, _S_gm_in, _eta_gm_in,
                    state_new.H_bathy.data,
                    _grid, _zc,
                    eos=_cfg_b.eos, eos_linear=_cfg_b.eos_linear,
                    eos_nemo_seos=_cfg_b.eos_nemo_seos,
                    mask=state.land_mask.data,
                    rho_0=_cfg_b.constants.rho_0,
                    g=_cfg_b.constants.g,
                    eos_depth=getattr(_cfg_b, "eos_depth", "insitu"),
                )
            # GM bolus THROUGH the FCT limiter (NEMO traadv): when the config
            # selects it (nemo_iso_lap + gm_bolus_advection="through_fct"), the
            # bolus transport is exported and added to the tracer advecting mass
            # flux instead of being applied as an in-operator centred flux.
            _want_bolus = (
                getattr(gm_cfg, "gm_bolus_advection", "centred") == "through_fct"
                and getattr(gm_cfg, "slope_scheme", "") == "nemo_iso_lap"
            )
            _gm_out = gm_redi_tracer_tendency_latlon(
                _T_gm_in, _S_gm_in, _eta_gm_in, state_new.H_bathy.data,
                _grid, _zc, gm_cfg,
                eos=_cfg_b.eos, eos_linear=_cfg_b.eos_linear,
                eos_nemo_seos=_cfg_b.eos_nemo_seos,
                mask=state.land_mask.data,
                u_mask=state.u_mask.data,
                v_mask=state.v_mask.data,
                rho_0=_cfg_b.constants.rho_0, g=_cfg_b.constants.g,
                # _cfg_b.omega (NOT _cfg_b.constants.Omega): the
                # same split as g's momentum-path/GM-Redi-path divergence
                # elsewhere in this function -- _cfg_b.omega is the
                # field dino_lat_lon_model_config actually threads
                # cfg.omega into (from_flat top-level field); constants.Omega
                # stays the unused NamedTuple default. See LatLonCGridOceanConfig
                # docstring / #1226.
                omega=_cfg_b.omega,
                kappa_gm_override=kappa_gm_override,
                kappa_redi_override=kappa_redi_override,
                kappa_redi_v_override=kappa_redi_v_override,
                density_jacobian=_gm_dens_jac,
                native_prd_jacobian=_gm_native_prd_J,
                native_prd_TS=_gm_native_prd_TS,
                native_slope_pn2=_gm_native_pn2,
                native_slope_e3w=_gm_native_e3w,
                native_slope_eta=state.eta.data,
                # ldf_eiv_trp precedes dynamics and consumes the same-stage
                # ldf_slp slopes, while the later tra_ldf tensor consumes Kmm
                # geometry -- that is the stpMLF reading, and on the leapfrog
                # lanes ``_eta_gm_in`` is ``state_new.eta.data``, so it is
                # unchanged there.  On the WS-RK3 lane there is exactly ONE
                # ldf_slp call per step and it is on Nbb (stprk3.F90:174), so
                # the bolus and the Redi tensor read the SAME before-state
                # slope field; an independent diff review found this operand
                # left on Naa while the comment above claimed the whole slope
                # operand set had moved.  MEASURED inert on GYRE: the card
                # resolves kappa_GM = 0.0 and gm_bolus_advection = "centred",
                # so the bolus is not even requested there.
                native_bolus_slope_eta=_eta_gm_in,
                # tra_ldf e3u/e3v use the step-entry Nnn SSH (Kmm).
                redi_flux_eta=state.eta.data,
                return_bolus_transport=_want_bolus,
                return_redi_diagnostics=((_return_tracer_process_trace or _return_ldf_diagnostic_trace) and self._nemo_ws_test_hooks.tracer_ldf_diagnostics is not None), return_redi_slope_diagnostics=((_return_tracer_process_trace or _return_ldf_diagnostic_trace) and (self._nemo_ws_test_hooks.tracer_ldf_diagnostics == "slope" or isinstance(self._nemo_ws_test_hooks.tracer_ldf_diagnostics, dict))), native_slope_nmln_override=(self._nemo_ws_test_hooks.tracer_ldf_diagnostics.get("nmln") if isinstance(self._nemo_ws_test_hooks.tracer_ldf_diagnostics, dict) else None),
                redi_face_thickness_override=(self._nemo_ws_test_hooks.tracer_ldf_diagnostics if isinstance(self._nemo_ws_test_hooks.tracer_ldf_diagnostics, tuple) else None),
                redi_divisor_thickness_override=(self._nemo_ws_test_hooks.tracer_ldf_diagnostics.get("divisor_thickness") if isinstance(self._nemo_ws_test_hooks.tracer_ldf_diagnostics, dict) else None),
                dt=dt,
                eos_depth=getattr(_cfg_b, "eos_depth", "insitu"),
            )
            if _want_bolus:
                dT_gm, dS_gm, _bolus = _gm_out
                if _bolus is not None:
                    mass_flux_u_tr, mass_flux_v_tr, w_baro_tr = (
                        add_bolus_to_advecting_flux(
                            _bolus, mass_flux_u, mass_flux_v,
                            u_mask_3d_tracer, v_mask_3d_tracer, _grid,
                            _zc,
                        )
                    )
            else:
                dT_gm, dS_gm, *_ldf_diag = _gm_out; _nemo_ws_ldf_diagnostics = _ldf_diag[0] if _ldf_diag else None
            if gm_cfg.implicit_K33:
                # Veros-faithful: K_33 (the vertical isoneutral diagonal ∝ S²) was
                # dropped from the explicit F_z above (implicit_K33=True); recompute
                # it from the SAME density/slopes as dT_gm and fold it into the
                # implicit vertical-diffusion solve below, so the stiff S²-enhanced
                # vertical mixing is applied backward-Euler exactly as in Veros.
                # ``_T_gm_in``/``_S_gm_in`` (Nbb under nemo_mlf, else T_mid):
                # under ``slope_positions="nemo_native"`` K33 recomputes N²
                # (_nemo_wpoint_e3w_wmask_n2) and the MLD ramp (_nemo_mld)
                # from these positional T,S even when ``density_jacobian`` is
                # provided -- so they MUST match the tendency call's tracer
                # source, or K33's slopes mix time levels against the
                # isoneutral flux they augment (adversarial-review CONFIRMED
                # finding, P1).
                k33_implicit = compute_isoneutral_K33_latlon(
                    _T_gm_in, _S_gm_in, _eta_gm_in,
                    state_new.H_bathy.data,
                    _grid, _zc, gm_cfg,
                    eos=_cfg_b.eos, eos_linear=_cfg_b.eos_linear,
                    eos_nemo_seos=_cfg_b.eos_nemo_seos,
                    mask=state.land_mask.data,
                    rho_0=_cfg_b.constants.rho_0, g=_cfg_b.constants.g,
                    omega=_cfg_b.omega,
                    kappa_redi_override=kappa_redi_override,
                    kappa_redi_v_override=kappa_redi_v_override,
                    density_jacobian=_gm_dens_jac,
                    native_prd_jacobian=_gm_native_prd_J,
                    native_prd_TS=_gm_native_prd_TS,
                    native_slope_pn2=_gm_native_pn2,
                    native_slope_e3w=_gm_native_e3w,
                    native_slope_eta=state.eta.data,
                    # #1226: the SAME wall masks the tendency dispatcher uses,
                    # so the nemo_native K33 slopes/masks are bit-identical to
                    # the explicit operator's (staircase-aware; the K33-side
                    # 2-D rebuild was flat-bottom-only).
                    u_mask=state.u_mask.data,
                    v_mask=state.v_mask.data,
                    dt=dt,
                    eos_depth=getattr(_cfg_b, "eos_depth", "insitu"),
                )
            if _ab2_advective:
                # AB2 "advective" scope: GM/Redi is a DISSIPATIVE (isoneutral +
                # skew) tracer term — Veros applies it at weight 1.0 to
                # ``tr[taup1]`` (core/isoneutral/diffusion.py), NOT inside the
                # AB2-extrapolated advection. Route its increment into the
                # weight-1.0 dissipative bucket instead of adding it to T_mid
                # (which is what the AB2 extrapolates).
                # active_3d (not the 2-D mask_3d): on a partial-cell coord the
                # GM/Redi tendency is nonzero at SUB-SEAFLOOR cells of wet
                # columns (the triad assembly masks 2-D only); writing it
                # there drifts dry S negative → sqrt(S<0) NaN in the gsw EOS
                # (global_4deg step-4 blowup). Veros masks with the 3-D maskT.
                # Pure z-star: active_3d == mask_3d ⇒ bit-identical.
                _diss_dT_incr = _diss_dT_incr + dt * dT_gm * active_3d
                _diss_dS_incr = _diss_dS_incr + dt * dS_gm * active_3d
            else:
                # active_3d for the same sub-seafloor reason as the diss bucket
                # above (bit-identical on pure z-star where active_3d==mask_3d).
                T_mid = T_mid + dt * dT_gm * active_3d
                S_mid = S_mid + dt * dS_gm * active_3d
            if eke_new is not None:
                state_new = state_new._replace(eke=eke_new)
            # Carry the EKE dissipation (Veros eke_diss_iw) for the prognostic-TKE
            # source's ONE-STEP-LAGGED recycling (only the 3-D path produces it).
            if eke_diss_new is not None:
                state_new = state_new._replace(eke_diss=eke_diss_new)

        if _cfg_b.tracer_advection == "som":
            # SOM (Prather 1986): Second Order Moments advection (#210).
            # Advects polynomial sub-cell distributions (mean + 9 moments)
            # via directional sweeps.  Near-zero spurious diapycnal mixing,
            # fully differentiable (no limiter).
            # Initialise moments: use existing Fields, or create from zeros.
            # Always produce Field output (not None → Field transition) so
            # the pytree structure is stable for jax.lax.scan.
            dims_mom = ("lat", "lon", "level", "moment")
            if state.T_som is not None:
                T_mom = state.T_som.data
                S_mom = state.S_som.data
            else:
                T_mom = jnp.zeros((*T_mid.shape, 9), dtype=T_mid.dtype)
                S_mom = jnp.zeros((*S_mid.shape, 9), dtype=S_mid.dtype)
                # Pre-create Fields on state_new so the output pytree
                # always has the same structure as the input.
                state_new = state_new._replace(
                    T_som=Field(data=T_mom, name="T_som",
                                dims=dims_mom, units=""),
                    S_som=Field(data=S_mom, name="S_som",
                                dims=dims_mom, units=""),
                )

            T_corrected, T_mom_new = som_advect_tracers(
                T_mid, T_mom, mass_flux_u_tr, mass_flux_v_tr, w_baro_tr,
                h_k_old, h_k_new, _grid, dt, mask,
            )
            S_corrected, S_mom_new = som_advect_tracers(
                S_mid, S_mom, mass_flux_u_tr, mass_flux_v_tr, w_baro_tr,
                h_k_old, h_k_new, _grid, dt, mask,
            )

            state_new = state_new._replace(
                T_som=state_new.T_som.replace(data=T_mom_new),
                S_som=state_new.S_som.replace(data=S_mom_new),
            )
        else:
            _adv = _cfg_b.tracer_advection
            _tti = _cfg_b.tracer_time_integrator
            _T_flux_div_cur = None
            _S_flux_div_cur = None

            # AB2: ensure pytree structure is stable for jax.lax.scan.
            # When the input state has None carry fields, pre-create
            # zero-filled Fields on state_new so the output pytree
            # always matches the input (same pattern as SOM moments).
            if _tti == "ab2" and state.T_flux_div_prev is None:
                from legoesm.core.field import Field as _Field
                _dims_fd = ("lat", "lon", "level")
                _zero_fd = jnp.zeros_like(T_mid)
                state_new = state_new._replace(
                    T_flux_div_prev=_Field(
                        data=_zero_fd, name="T_flux_div_prev",
                        dims=_dims_fd, units="m/s"),
                    S_flux_div_prev=_Field(
                        data=_zero_fd, name="S_flux_div_prev",
                        dims=_dims_fd, units="m/s"),
                )

            # Wall tracer BC (#480): the flux-form reconstructions read the
            # RAW tracer, whose land cells hold the masked fill value (T=0)
            # — a cold cell that the WIDE WENO stencil at the first wet
            # faces pulls in, manufacturing a spurious near-wall tracer
            # front.  Under a 2Δx-in-lon v perturbation this front sources
            # an un-dissipatable grid mode at free-slip walls (the §5
            # eddy-permitting blow-up).  Oceananigans' clean grid-edge wall
            # has no such cold cell.  Faithful cure: zero-gradient (Neumann)
            # fill the tracer over the DEAD cells INSIDE the reconstruction (see
            # _compute_advection_flux_div) so the stencil sees a flat extension
            # = the physical no-flux insulating wall, while the flux-form
            # UPDATE / gating below keeps the ORIGINAL dead-cell value.  Use the
            # per-level ``active_3d`` (= is_active) mask, NOT the 2D surface
            # land_mask, so the fill also cleans TOPOGRAPHIC-STEP dead cells in
            # partial-cell runs (flat-bottom: active_3d is (n_lat,n_lon,1) and
            # the fill is bit-identical to the 2D-mask path).
            _wall_fill_mask = (
                active_3d if getattr(_cfg_b, "tracer_wall_neumann_fill", True)
                else None
            )

            # T+S pair fast path: ONE fused horizontal reconstruction +
            # N-S pad per stage for both tracers (level-axis stack; see
            # compute_advection_flux_div_pair).  Bit-identical to the
            # historical per-tracer calls; non-separable schemes fall
            # back to two single-tracer calls inside the pair helpers.
            # NEMO key_linssh: top-cell concentration/dilution flux (static
            # coordinate flag; see _compute_advection_flux_div).
            _linssh = getattr(_zc, "linear_free_surface", False)
            # Leap-frog FCT monotonicity base (Kbb).  The FCT-limited advective
            # increment is applied to the BEFORE level; carry the before tracer
            # through the SAME pre-advection physics increment as ``T_mid`` so
            # the FCT bounds/low-order match the actual after-state base:
            #   base = T_before + (T_mid − T_now).
            # ``None`` (FE/AB2) ⇒ byte-identical.  RK3-under-leapfrog is not a
            # configured path, so it keeps the NOW base (unchanged).
            if _fct_tracer_before is not None:
                _T_adv_before = _fct_tracer_before[0] + (T_mid - state.T.data)
                _S_adv_before = _fct_tracer_before[1] + (S_mid - state.S.data)
            else:
                _T_adv_before = _S_adv_before = None
            if _tti == "rk3":
                T_corrected, S_corrected = _ssp_rk3_tracer_pair_step(
                    T_mid, S_mid, _adv,
                    mass_flux_u_tr, mass_flux_v_tr, w_baro_tr,
                    h_k_old, h_k_new, h_u_old, h_v_old,
                    _grid, dt, active_3d, recon_fill_mask=_wall_fill_mask,
                    linssh_top_flux=_linssh,
                )
                _pair_divs = (None, None)
            elif _tti == "rk3_ws":
                _ws_stage_source_rates = _stage_source_rates
                if _cfg_b.gm_redi is not None:
                    _ws_stage_source_rates = (
                        _stage_source_rates[0],
                        _stage_source_rates[1],
                        (
                            _stage_source_rates[2][0] + dT_gm * active_3d,
                            _stage_source_rates[2][1] + dS_gm * active_3d,
                        ),
                    )
                _bbl_context = None
                if ((_cfg_b.bbl_adv_option == 2
                     or _cfg_b.bbl_diffusive_option == 1)
                        and not self._nemo_ws_test_hooks.disable_bbl):
                    from legoesm.ocean.physics.bbl_adv import (
                        bbl_static_geometry,
                        nemo_bbl_diffusive_geometry,
                        nemo_bbl_static_geometry,
                    )
                    _h_ref = jnp.asarray(_zc.h_partial)
                    if _h_ref.ndim == 1:
                        _h_ref = jnp.broadcast_to(_h_ref, h_k_old.shape)
                    _nemo_bbl_geometry = not (
                        self._nemo_ws_test_hooks.legacy_bbl_partial_geometry)
                    _bbl_geom = None
                    if _cfg_b.bbl_adv_option == 2 and _nemo_bbl_geometry:
                        _gdept0 = getattr(_zc, "nemo_gdept_0", None)
                        _e3u0 = getattr(_zc, "nemo_bbl_e3u_0", None)
                        _e3v0 = getattr(_zc, "nemo_bbl_e3v_0", None)
                        if _gdept0 is None or _e3u0 is None or _e3v0 is None:
                            raise ValueError(
                                "NEMO BBL reference geometry requires "
                                "nemo_gdept_0 and exact unmasked "
                                "nemo_bbl_e3u_0/nemo_bbl_e3v_0 operands")
                        _bbl_geom = nemo_bbl_static_geometry(
                            _h_ref, state.land_mask.data,
                            _gdept0, _e3u0, _e3v0)
                    elif _cfg_b.bbl_adv_option == 2:
                        _bbl_geom = bbl_static_geometry(
                            _h_ref, state.land_mask.data)
                    _diffusive_geom = None
                    if _cfg_b.bbl_diffusive_option == 1:
                        _gdept0 = getattr(_zc, "nemo_gdept_0", None)
                        _e3u0 = getattr(_zc, "nemo_bbl_e3u_0", None)
                        _e3v0 = getattr(_zc, "nemo_bbl_e3v_0", None)
                        _raw_bbl = getattr(_zc, "nemo_een_barotropic", None)
                        if (any(value is None
                                for value in (_gdept0, _e3u0, _e3v0))
                                or _raw_bbl is None):
                            raise ValueError(
                                "NEMO diffusive BBL requires exact gdept_0, "
                                "e3u_0/e3v_0, metrics, and U/V masks")
                        _diffusive_geom = nemo_bbl_diffusive_geometry(
                            _h_ref, state.land_mask.data,
                            _gdept0, _e3u0, _e3v0,
                            _raw_bbl.e1u, _raw_bbl.e2u,
                            _raw_bbl.e1v, _raw_bbl.e2v,
                            _raw_bbl.umask, _raw_bbl.vmask,
                            aht_m2_s=_cfg_b.bbl_aht_m2_s, grid=_grid,
                        )
                    _bbl_context = (
                        _cfg_b.bbl_adv_option,
                        _cfg_b.bbl_diffusive_option,
                        _bbl_geom,
                        _diffusive_geom,
                        jnp.asarray(_grid.area_T),
                        jnp.asarray(_grid.dy_u)[:, 1:-1],
                        jnp.asarray(_grid.dx_v)[1:-1, :],
                        _cfg_b.bbl_gamma_s,
                        _cfg_b.bbl_aht_m2_s,
                        _cfg_b.rho_0,
                        _nemo_bbl_geometry,
                        _grid,
                        _cfg_b.eos,
                    )
                _ws_tracer_result = _nemo_ws_rk3_tracer_pair_step(
                    # NEMO's stage ladder always restarts from ts(Kbb).
                    # The explicit non-advective tendency is supplied below
                    # as the stage-3 Krhs source, not pre-applied to this base.
                    state.T.data, state.S.data, _adv,
                    mass_flux_u_tr, mass_flux_v_tr, w_baro_tr,
                    h_k_old, h_k_new, h_u_old, h_v_old,
                    _grid, dt, active_3d, recon_fill_mask=_wall_fill_mask,
                    linssh_top_flux=_linssh,
                    stage_transport_geometry=(
                        tuple(
                            (mf_u, mf_v, jnp.zeros_like(w_stage), h_stage,
                             hu_stage, hv_stage, jnp.zeros_like(wi_stage),
                             zfu_stage, zfv_stage, corrected_u, corrected_v)
                            for (mf_u, mf_v, w_stage, h_stage, hu_stage,
                                 hv_stage, wi_stage, zfu_stage, zfv_stage,
                                 corrected_u, corrected_v)
                            in _nemo_ws_stage_transport_geometry
                        )
                        if self._nemo_ws_test_hooks.disable_tracer_vertical_transport
                        else _nemo_ws_stage_transport_geometry
                    ),
                    fct_low_order_predictor=(
                        "nemo_rk3_two_step"
                        if self._nemo_ws_test_hooks.two_step_fct_predictor
                        else "one_step"),
                    stage_source_rates=_ws_stage_source_rates,
                    bbl_context=_bbl_context,
                    # One stage ladder: stages 1-2 were advanced by the
                    # momentum program on the same Kmm transports as the
                    # stage-2/3 hpg operands; stage 3 finishes on them.  The
                    # vertical-transport ablation hook rebuilds its own ladder.
                    resume=(
                        None
                        if (_nemo_ws_stage_tracers is None
                            or self._nemo_ws_test_hooks
                            .disable_tracer_vertical_transport)
                        else (2, *_nemo_ws_stage_tracers)),
                    stage3_advection_content_override=(
                        self._nemo_ws_test_hooks
                        .stage3_advection_content_override),
                    return_final_content=True,
                    return_fct_activity=(
                        _return_tracer_process_trace
                        and self._nemo_ws_test_hooks
                        .tracer_process_branch_activity),
                )
                if (_return_tracer_process_trace
                        and self._nemo_ws_test_hooks
                        .tracer_process_branch_activity):
                    (T_corrected, S_corrected,
                     _nemo_ws_content_T, _nemo_ws_content_S,
                     _nemo_ws_advection_content_T,
                     _nemo_ws_advection_content_S,
                     _nemo_ws_fct_activity) = _ws_tracer_result
                else:
                    (T_corrected, S_corrected,
                     _nemo_ws_content_T, _nemo_ws_content_S,
                     _nemo_ws_advection_content_T,
                     _nemo_ws_advection_content_S) = _ws_tracer_result
                    _nemo_ws_fct_activity = None
                _nemo_ws_tracer_content_rhs = (
                    _nemo_ws_content_T, _nemo_ws_content_S)
                _nemo_ws_advection_content_rhs = (
                    _nemo_ws_advection_content_T,
                    _nemo_ws_advection_content_S)
                if _return_tracer_process_trace:
                    if (_nemo_ws_process_qco is None
                            or _nemo_ws_process_surface_rate is None
                            or _nemo_ws_process_qsr_rate is None
                            or _cfg_b.gm_redi is None):
                        raise ValueError(
                            "tracer process trace requires the resolved "
                            "NEMO GYRE QSR and GM/Redi stage-3 program")
                    _qbb, _qmm, _qaa, _hbb, _hmm, _haa = (
                        _nemo_ws_process_qco)
                    _process_adv = _nemo_ws_advection_content_T
                    _process_sbc = (
                        _process_adv + dt * _hmm
                        * _nemo_ws_process_surface_rate)
                    _process_qsr = (
                        _process_sbc + dt * _hmm
                        * _nemo_ws_process_qsr_rate)
                    _process_ldf = (
                        _process_qsr + dt * _hmm
                        * (dT_gm * active_3d))
                    _h_safe_process = jnp.maximum(_haa, 1.0e-10)
                    _nemo_ws_process_boundaries = (
                        _hbb * state.T.data / _h_safe_process,
                        _process_adv / _h_safe_process,
                        _process_sbc / _h_safe_process,
                        _process_qsr / _h_safe_process,
                        _process_ldf / _h_safe_process,
                        _nemo_ws_content_T / _h_safe_process,
                    )
                _pair_divs = (None, None)
            else:
                # store_salt_flux capture (static config bool): ask the pair
                # helper for tracer B's (= SALT's) horizontal face flux pair
                # -- the SAME arrays whose divergence updates S below, not a
                # recomputation.  Constructor guards restrict the flag to the
                # euler integrator and exposed-flux schemes, so this branch
                # is the only advection lane it can reach.
                _cap_salt = bool(getattr(_cfg_b, "store_salt_flux",
                                         False))
                if _cap_salt:
                    _pair_a, _pair_b, (_sf_u3, _sf_v3) = (
                        compute_advection_flux_div_pair(
                            T_mid, S_mid, _adv,
                            mass_flux_u_tr, mass_flux_v_tr, w_baro_tr,
                            h_k_old, h_u_old, h_v_old, _grid, dt,
                            recon_fill_mask=_wall_fill_mask,
                            linssh_top_flux=_linssh,
                            tr_a_before=_T_adv_before,
                            tr_b_before=_S_adv_before,
                            return_b_h_fluxes=True,
                        ))
                    _pair_divs = (_pair_a, _pair_b)
                    # Column integral NOW, while the 3-D pair is live: the
                    # state stores only the 2-D vertical sum (see the state
                    # docstring for the 145 MB-vs-2 MB decision).
                    _salt_flux_u2 = _sf_u3.sum(axis=-1)
                    _salt_flux_v2 = _sf_v3.sum(axis=-1)
                else:
                    _pair_divs = compute_advection_flux_div_pair(
                        T_mid, S_mid, _adv,
                        mass_flux_u_tr, mass_flux_v_tr, w_baro_tr,
                        h_k_old, h_u_old, h_v_old, _grid, dt,
                        recon_fill_mask=_wall_fill_mask,
                        linssh_top_flux=_linssh,
                        tr_a_before=_T_adv_before, tr_b_before=_S_adv_before,
                    )

            for tr_name in ['T', 'S']:
                tr = T_mid if tr_name == 'T' else S_mid

                if _tti in ("rk3", "rk3_ws"):
                    # Computed by the pair step above.
                    continue
                else:
                    div_hut, vert_flux_div = (
                        _pair_divs[0] if tr_name == 'T' else _pair_divs[1]
                    )
                    total_flux_div = div_hut + vert_flux_div

                    if _tti == "ab2":
                        # Adams-Bashforth 2: extrapolate flux divergence
                        # F^{n+1/2} = (3/2+eps)*F^n - (1/2+eps)*F^{n-1}
                        eps = _cfg_b.ab2_epsilon
                        prev_field = (state.T_flux_div_prev
                                      if tr_name == 'T'
                                      else state.S_flux_div_prev)
                        if _pflow is not None:
                            # Prescribed-flow lever (codex finding 1): the
                            # advective flux-div extrapolation is CIRCULATION
                            # HISTORY — a carried F^{n-1} from a pre-lever /
                            # poisoned regime leaks stale advection, because
                            # ab2_blend(0, fd_prev, eps) is NONZERO even when
                            # the pinned flux-div is exactly zero.  Consume
                            # NOTHING from the carry: use the current PINNED
                            # flux-div directly ("zero" → identically zero, no
                            # advection at all; "frozen" → the entry-flow
                            # flux-div, no extrapolation across regimes).  The
                            # carry STORED below is this same current value
                            # (zeros for "zero"), so it stays scan-stable
                            # (Field→Field) while never being read.  The T/S
                            # PHYSICS AB2 increments (T_incr_prev, outer
                            # integrator) are untouched — only the ADVECTION
                            # extrapolation is disabled.
                            effective_fd = total_flux_div
                        elif prev_field is not None:
                            fd_prev = prev_field.data
                            # (#517 item 8: shared ab2_blend; eps verbatim
                            # → bit-identical.)
                            effective_fd = ab2_blend(total_flux_div, fd_prev,
                                                     eps)
                        else:
                            # First step: fall back to Euler
                            effective_fd = total_flux_div
                        hT_new = h_k_old * tr - dt * effective_fd
                    else:
                        # Forward Euler (default)
                        hT_new = h_k_old * tr - dt * total_flux_div

                    tr_new = hT_new / jnp.maximum(h_k_new, 1e-10)
                    tr_new = jnp.where(active_3d > 0.5, tr_new, tr)

                    # Store current flux divergence for AB2 carry
                    if _tti == "ab2":
                        if tr_name == 'T':
                            _T_flux_div_cur = total_flux_div
                        else:
                            _S_flux_div_cur = total_flux_div

                if tr_name == 'T':
                    T_corrected = tr_new
                else:
                    S_corrected = tr_new

            # Persist AB2 previous flux divergence on state.  Under the
            # prescribed-flow lever this stores the CURRENT pinned flux-div
            # (exactly zeros for "zero": the mass fluxes and diagnosed w
            # vanish), keeping the carry pytree scan-stable; the consumption
            # branch above never reads it while the lever is on.
            if _tti == "ab2":
                from legoesm.core.field import Field as _Field
                _dims_fd = ("lat", "lon", "level")
                state_new = state_new._replace(
                    T_flux_div_prev=_Field(
                        data=_T_flux_div_cur, name="T_flux_div_prev",
                        dims=_dims_fd, units="m/s"),
                    S_flux_div_prev=_Field(
                        data=_S_flux_div_cur, name="S_flux_div_prev",
                        dims=_dims_fd, units="m/s"),
                )

        # Store the current barotropic slow forcing as next-step prev for the
        # AB2 time-centering.  Pytree-stable: the apply block above REQUIRES the
        # prev to be a seeded Field when the flag is on (it raises on None), so
        # this storage is always Field→Field across scan iterations.  ``_F_slow_
        # *_cur`` is the CURRENT (non-extrapolated) value, captured BEFORE the
        # du_diss depth-mean fold (ab2_scope="advective" is rejected upstream, so
        # under the supported ab2_scope="total" path du_diss is None and there is
        # nothing extra to fold).
        if getattr(_cfg_b.barotropic, "barotropic_slow_forcing_ab2", False):
            from legoesm.core.field import Field as _Field_fs
            # Prescribed-flow lever: store ZEROS (pytree structure unchanged —
            # still Field→Field across scan iterations).  The barotropic solve
            # is discarded under the lever, so carrying its slow-forcing
            # history would AB2-extrapolate discarded-momentum state (codex
            # finding 2 — stale carries must be neutralized consistently).
            _fs_u_store = (_F_slow_u_cur if _pflow is None
                           else jnp.zeros_like(_F_slow_u_cur))
            _fs_v_store = (_F_slow_v_cur if _pflow is None
                           else jnp.zeros_like(_F_slow_v_cur))
            state_new = state_new._replace(
                F_slow_u_prev=_Field_fs(
                    data=_fs_u_store, name="F_slow_u_prev",
                    dims=("lat", "lon_u"), units="m/s^2"),
                F_slow_v_prev=_Field_fs(
                    data=_fs_v_store, name="F_slow_v_prev",
                    dims=("lat_v", "lon"), units="m/s^2"),
            )

        # Include vertical velocity diagnostic in state
        # w_baro has shape (..., nlev+1) on half levels, interpolate to full levels (..., nlev)
        w_full = 0.5 * (w_baro[..., :-1] + w_baro[..., 1:])  # Average adjacent half levels
        w_field = state.w.replace(data=w_full, name="w")

        state_new = state_new._replace(
            T=state_new.T.replace(data=T_corrected),
            S=state_new.S.replace(data=S_corrected),
            w=w_field,
        )

        # #1442: keep the tracer-advecting mass fluxes instead of discarding
        # them.  PURE DIAGNOSTIC -- read-only capture of values this step
        # already computed; nothing above depends on the branch, so the
        # trajectory is bit-identical with the flag off or on.
        #
        # ``mass_flux_u_tr``/``mass_flux_v_tr`` (NOT ``mass_flux_u``/``_v``) are
        # what lines ~3961/3965 actually advect T and S with: they start as
        # ``h_u_old * u_corrected`` -- i.e. WITH the barotropic transport
        # correction ``(Hu_avg - Hu_3d)/H_u_old`` that never reaches
        # ``state.u`` -- and are REPLACED by the bolus-inclusive flux under the
        # EXACT condition ``_want_bolus`` tests above, namely BOTH
        # ``gm_redi.gm_bolus_advection == "through_fct"`` AND
        # ``gm_redi.slope_scheme == "nemo_iso_lap"`` (the default "centred"
        # bolus is an in-operator flux and never enters this pair; no other
        # slope scheme exports a bolus transport at all).  Storing the ``_tr``
        # pair therefore closes the barotropic omission always, and the
        # GM-bolus omission in that configuration.
        #
        # Static Python bool on a config leaf, so this is a compile-time
        # branch (the CLAUDE.md feature-gating exception): only one side is
        # ever traced and the pytree structure is fixed for the whole run.
        #
        # A MATCHED TRIPLE: ``w_baro_tr`` is stored alongside the horizontal
        # pair.  ``state.w`` above is built from the BASE ``w_baro`` (it is the
        # momentum/continuity/eta vertical velocity and must not change), so
        # under through-FCT GM the stored pair and ``state.w`` are NOT
        # consistent -- storing the pair's own vertical partner is what closes
        # that (codex YELLOW 9).  With GM off, ``w_baro_tr is w_baro``.
        #
        # Metadata comes from the SHARED ``mass_flux_fields`` constructor that
        # ``seed_mass_flux_carry`` also uses, so the scan carry's treedef
        # (Field name/dims/units are pytree AUX data) matches this exactly.
        if _cfg_b.store_mass_flux:
            _mfu_field, _mfv_field, _mfw_field = mass_flux_fields(
                state.u, state.v, state.w,
                mass_flux_u_tr, mass_flux_v_tr, w_baro_tr)
            state_new = state_new._replace(mass_flux_u=_mfu_field,
                                           mass_flux_v=_mfv_field,
                                           mass_flux_w=_mfw_field)

        # store_salt_flux: the column-integrated advective SALT flux pair the
        # S update above was evaluated with (gateway exact-salt instrument).
        # Same pure-diagnostic contract as the mass triple: read-only capture
        # of arrays this step already computed; static config bool; metadata
        # through the SHARED salt_flux_fields constructor so the scan-carry
        # treedef matches seed_salt_flux_carry exactly.  The constructor
        # guards (euler + exposed-flux scheme + not SOM) make the capture
        # branch above the only lane, so _salt_flux_u2/_v2 are always bound
        # here when the flag is on.
        if getattr(_cfg_b, "store_salt_flux", False):
            _sfu_field, _sfv_field = salt_flux_fields(
                state.u, state.v, _salt_flux_u2, _salt_flux_v2)
            state_new = state_new._replace(salt_flux_u_int=_sfu_field,
                                           salt_flux_v_int=_sfv_field)

        # 8. Freshwater forcing (virtual salt flux only)
        #
        # The freshwater eta tendency (F_fw_eta) is now applied inside
        # the barotropic continuity equation (via F_slow_eta), so no
        # post-hoc eta correction is needed.  Only the virtual salt
        # flux remains here, applied to the top layer of S.
        # `real_freshwater` (NEMO variable-volume convention): the eta/volume
        # channel above ALREADY represents dilution -- the z-star tracer step
        # hS_new = h_old*S - dt*div  then  S_new = hS_new/h_new preserves h*S
        # while the column stretches.  Adding a virtual salt flux on top of
        # that is a SEPARATE salt-content source that NEMO does not have; it
        # was measured at +10.109 psu.m of spurious Arctic salt over 60 days
        # (docs/dev-notes/ocean_real_freshwater_design.md).  So skip ONLY this
        # block; the genuine surface_forcing.salt_flux pathway is untouched.
        if (freshwater is not None
                and _cfg_b.freshwater_closure
                not in ("none", "real_freshwater")):
            dz_0 = h_k_new[..., 0]
            _S_dtype = state_new.S.data.dtype
            # Salinity entering the virtual-salt closure: the fixed scalar
            # S_ref (legacy, bit-identical) or the LOCAL top-cell salinity —
            # NEMO's tra_sbc convention (sfx = emp * sss).  Static config
            # gate; unknown values raise at model construction.  The local
            # field is the post-advection (Now) salinity, column-constant
            # for the runoff-spread channel.
            if getattr(_cfg_b, "freshwater_salinity", "s_ref") == "local":
                _S_fw = state_new.S.data[..., 0].astype(_S_dtype)
            else:
                _S_fw = _cfg_b.S_ref
            from legoesm.ocean.freshwater import resolve_runoff_spread_arg
            _spread_arg = resolve_runoff_spread_arg(_cfg_b)
            if _spread_arg is not None:
                # NEMO-style runoff depth spreading (rn_dep_max): the runoff
                # channel dilutes the top spread depth — a flat scalar
                # (runoff_depth_spread_m) OR the per-cell ln_rnf_depth_ini map
                # (runoff_depth_spread_map); all other channels stay at the
                # top cell.  Column-integral salt tendency identical to the
                # legacy closure (conservation unchanged).  Static config gate
                # -> legacy path untraced.  The map/scalar selection +
                # mutual-exclusion guard is shared with the MPAS Voronoi core
                # via resolve_runoff_spread_arg (no per-grid copy-paste).
                from legoesm.ocean.freshwater import (
                    runoff_spread_virtual_salt_tendency_3d,
                )
                dS_fw_3d = runoff_spread_virtual_salt_tendency_3d(
                    freshwater, _S_fw, h_k_new, _cfg_b.rho_0,
                    mask,
                    runoff_spread_m=_spread_arg,
                    area=_grid.area,
                    normalize=bool(getattr(_cfg_b,
                                           "normalize_freshwater", False)),
                )
                S_fw = state_new.S.data + (
                    dt * dS_fw_3d * mask[..., None]).astype(_S_dtype)
            elif getattr(_cfg_b, "normalize_freshwater", False):
                # Global-salt-conserving virtual salt: remove the area-mean of the
                # net freshwater (the OMIP correction) so an unbalanced ∮(P-E+R)
                # does not drift mean salinity.  Shared with the MPAS path.
                from legoesm.ocean.freshwater import normalized_virtual_salt_flux
                dS_fw = normalized_virtual_salt_flux(
                    freshwater, _S_fw, dz_0, _cfg_b.rho_0,
                    _grid.area, mask,
                )
                S_fw = state_new.S.data.at[..., 0].add(
                    (dt * dS_fw * mask).astype(_S_dtype),
                )
            else:
                dS_fw = virtual_salt_flux(
                    freshwater, S_ref=_S_fw, dz_0=dz_0, rho_0=_cfg_b.rho_0,
                )
                # Cast the freshwater contribution to S's dtype so the
                # scatter add does not silently widen on x64 mode (the
                # freshwater struct is built at JAX-default precision in
                # init helpers, which can be f64 while S runs at the
                # storage policy's f32).
                S_fw = state_new.S.data.at[..., 0].add(
                    (dt * dS_fw * mask).astype(_S_dtype),
                )
            # Temperature twin (2026-09-05): the surface heat flux already
            # carries the rain/evap/restoring heat content, so dilute T by
            # the same water (NEMO linear-free-surface trasbc emp*sst term).
            from legoesm.ocean.freshwater import virtual_closure_temperature_twin
            _dT_twin = virtual_closure_temperature_twin(
                freshwater, state_new.T.data[..., 0], dz_0, _cfg_b.rho_0, mask)
            state_new = state_new._replace(
                S=state_new.S.replace(data=S_fw),
                T=state_new.T.replace(data=state_new.T.data.at[..., 0].add(
                    (dt * _dT_twin).astype(state_new.T.data.dtype))),
            )
        elif (freshwater is not None
                and _cfg_b.freshwater_closure == "real_freshwater"
                and _F_fw_rate_now is not None):
            # 8-real. Surface dilution of the volume closure (2026-09-05):
            # the water entered the TOP cell, the z-star step spread it over
            # the column; pair that with the downward transport of the
            # resident water so the top cell dilutes by -S_1 F/(rho h_1) and
            # the layers below keep S and T (NEMO vvl: sshwzv/traadv, no
            # emp*sss salt term).  Same NOW rate as the eta channel
            # (normalised, restoring included); the ice SALT flux stays on
            # surface_forcing.salt_flux.  Shared helper (MPAS core too).
            from legoesm.ocean.freshwater import (
                real_freshwater_dilution_tendencies,
                real_freshwater_entry,
                resolve_runoff_spread_arg,
            )
            _S_dtype = state_new.S.data.dtype
            # NEMO sbc_rnf_div: river water enters every level down to
            # h_rnf (same depth map the virtual closure spreads over); the
            # remaining channels enter at the surface.  Static config gate.
            _F_entry, _entry_heat = real_freshwater_entry(
                freshwater, _F_fw_rate_now, h_k_new, mask, _cfg_b.rho_0,
                state_new.T.data,
                runoff_spread_m=resolve_runoff_spread_arg(_cfg_b))
            _dS_dil, _dT_dil = real_freshwater_dilution_tendencies(
                _F_fw_rate_now, state_new.S.data, state_new.T.data,
                h_k_new, mask, F_entry=_F_entry, entry_heat=_entry_heat)
            state_new = state_new._replace(
                S=state_new.S.replace(
                    data=state_new.S.data + (dt * _dS_dil).astype(_S_dtype)),
                T=state_new.T.replace(
                    data=state_new.T.data + (dt * _dT_dil).astype(
                        state_new.T.data.dtype)),
            )

        # 8a'. AB2 "advective" scope: apply the weight-1.0 DISSIPATIVE increment.
        # On the FORWARD-EULER ``step()`` path (``_apply_implicit_vmix=True``)
        # the dissipative increment (momentum lateral friction + bottom drag;
        # tracer lateral diffusion + GM/Redi) is added INLINE here at weight 1.0
        # to the post-advection / post-barotropic explicit state — exactly its
        # placement under "total", just NOT folded into the AB2'd bucket (the FE
        # path doesn't AB2, so "total" and "advective" agree to round-off here;
        # the split MATTERS only for the AB2 outer integrator). On the AB2 path
        # (``_apply_implicit_vmix=False``) the increment is NOT applied here; it
        # is returned for ``_ab2_step`` to add at weight 1.0 after the AB2
        # extrapolation of the advective increment. ``_diss_*_incr`` is ``None``
        # under "total" ⇒ this block is skipped ⇒ bit-identical.
        if _diss_dT_incr is not None and _apply_implicit_vmix:
            # MOMENTUM: only the BAROCLINIC deviation — the increment's
            # depth-mean already forced the barotropic solve via F_slow (the
            # Veros uloc structure; see the F_slow diss fold above). Adding
            # the raw increment here would double-count the depth-mean and,
            # under the rigid lid, have it discarded next step anyway (the
            # review-caught routing bug).
            _du_bc = _diss_du_incr - (
                jnp.sum(_diss_du_incr * h_u_pre, axis=-1, keepdims=True)
                / jnp.maximum(jnp.sum(h_u_pre, axis=-1, keepdims=True), 1e-10))
            _dv_bc = _diss_dv_incr - (
                jnp.sum(_diss_dv_incr * h_v_pre, axis=-1, keepdims=True)
                / jnp.maximum(jnp.sum(h_v_pre, axis=-1, keepdims=True), 1e-10))
            u_diss_new = (state_new.u.data + _du_bc) * u_mask_3d
            u_diss_new = u_diss_new.at[:, -1].set(u_diss_new[:, 0])
            state_new = state_new._replace(
                T=state_new.T.replace(
                    data=(state_new.T.data + _diss_dT_incr) * mask_3d),
                S=state_new.S.replace(
                    data=(state_new.S.data + _diss_dS_incr) * mask_3d),
                u=state_new.u.replace(data=u_diss_new),
                v=state_new.v.replace(
                    data=(state_new.v.data + _dv_bc) * v_mask_3d),
            )

        _nemo_ws_pre_implicit_state = (
            state_new
            if self._nemo_ws_test_hooks.expose_pre_implicit_state else None)
        if self._nemo_ws_test_hooks.expose_pre_implicit_content:
            if _nemo_ws_tracer_content_rhs is None:
                raise ValueError(
                    "expose_pre_implicit_content requires a thickness-form "
                    "NEMO tracer content RHS")
            _content_diag_T, _content_diag_S = _nemo_ws_tracer_content_rhs
            _nemo_ws_pre_implicit_state = state_new._replace(
                T=state_new.T.replace(data=_content_diag_T),
                S=state_new.S.replace(data=_content_diag_S),
            )
        if self._nemo_ws_test_hooks.expose_stage3_advection_content:
            if _nemo_ws_advection_content_rhs is None:
                raise ValueError(
                    "expose_stage3_advection_content requires the NEMO "
                    "WS-RK3 tracer program")
            _adv_content_T, _adv_content_S = _nemo_ws_advection_content_rhs
            _nemo_ws_pre_implicit_state = state_new._replace(
                T=state_new.T.replace(data=_adv_content_T),
                S=state_new.S.replace(data=_adv_content_S),
            )
        _pre_zdf_override = (
            self._nemo_ws_test_hooks.pre_implicit_tracer_override)
        if _pre_zdf_override is not None:
            _pre_zdf_T, _pre_zdf_S = _pre_zdf_override
            state_new = state_new._replace(
                T=state_new.T.replace(data=_pre_zdf_T),
                S=state_new.S.replace(data=_pre_zdf_S),
            )
        _pre_zdf_content_override = (
            self._nemo_ws_test_hooks.pre_implicit_tracer_content_override)
        if _pre_zdf_content_override is not None:
            _nemo_ws_tracer_content_rhs = _pre_zdf_content_override

        # 8b. Implicit (backward-Euler) vertical mixing for u, v, T, S.
        #
        # When this branch is active, the PE tendency function has
        # skipped its explicit ``config.A_v`` block and every vertical-
        # mixing / convection scheme has been called with
        # ``apply_diffusion=False`` (so they only contributed K_v/A_v
        # *profiles* — and KPP non-local fluxes — to the explicit
        # update).  We now apply the full diffusion implicitly so the
        # stiff ``K_conv = 1 m²/s`` and surface-BL diffusivities are
        # not constrained by the explicit CFL limit
        # ``dt < dz² / (2K)`` — see issue #204.
        #
        # The solve uses zero-flux boundary conditions at the surface
        # and bottom and is split-stepped (Lie splitting, 1st-order)
        # after tracer advection, GM/Redi, and the freshwater virtual
        # salt flux — matching MOM6's diabatic-process ordering.
        # NEMO stprk3_stg:440 zub correction (nemo_stage_mean_imposition).
        # The WS stage ladder hands its own correction closure down here
        # (``_ws_stage3_correction``) and it needs no capture: its target is
        # the prognostic uu_b(Kaa) NEMO reads, not a mean measured before the
        # solve.  Every other path still captures the pre-solve depth mean so
        # it can be re-imposed after a solve that would otherwise shift it.
        _impose_mean = (
            getattr(_cfg_b.barotropic, "nemo_stage_mean_imposition", False)
            and _apply_implicit_vmix)
        if _impose_mean and _ws_stage3_correction is None:
            _u_mean_baro, _v_mean_baro = self._fixed_depth_means(state_new, z_coord=z_coord, config=config, grid=grid)

        tke_new = None
        if _cfg_b.implicit_vertical_mixing and _apply_implicit_vmix:
            if self._tke_prognostic_active():
                # PROGNOSTIC TKE: seed from the carried state.tke, assemble the
                # energy-recycling source forc (eke_diss_iw from THIS step's EKE
                # update + K_diss_bot from this step's tendency), run ONE
                # backward-Euler TKE step (dt = dt_mom) inside the implicit solve,
                # and carry the updated TKE forward.
                _tke_old = state.tke.data if state.tke is not None else None
                # eke_diss is read from state_new (THIS step's EKE update, which
                # runs at step 7/GM-Redi BEFORE this implicit-mixing TKE solve) —
                # matching Veros's same-step eke→tke ordering (veros.py:277,285),
                # so there is NO lag in the synchronous (non-AB2) path. Falls back
                # to the carried state.eke_diss when state_new has none yet.
                _tke_source = self._assemble_tke_source(state, state_new, tend, z_coord=z_coord, config=config)
                # NEMO eosbn2 Nnow sequencing: sample the diffusivity-stage
                # N² on the STEP-ENTRY (before-advection) T/S when the flag is
                # set. ``state`` here is the step-entry state (never rebound;
                # ``state_new`` is the working copy). None ⇒ BIT-IDENTICAL.
                _n2_tracers = self._n2_before_advection_tracers(state, z_coord=z_coord, config=config)
                _n2_tracers_before = self._n2_nemo_before_tracers(state, z_coord=z_coord, config=config)
                _tke_result = self._apply_implicit_vertical_mixing(
                    state_new, dt, surface_forcing,
                    K_v_phys=tend.K_v, A_v_phys=tend.A_v,
                    K33_iso=k33_implicit, dt_mom=dt_mom,
                    surface_tracer_forcing=tend.surface_tracer_forcing,
                    tracer_source=tend.tracer_source,
                    tke_old=_tke_old, tke_source=_tke_source, return_tke=True,
                    return_tke_entry=_return_live_stage_operands, return_tracer_solve_trace=_return_vertical_solve_trace,
                    grid=_grid, n2_tracers=_n2_tracers,
                    n2_tracers_before=_n2_tracers_before,
                    tke_n2_bundle=_tke_n2_bundle,
                    # NEMO e3w(Kmm) divisor: state_new is the post-update
                    # AFTER state; state.eta is NOW. Read only under the NEMO
                    # identity (zdf_implicit_solver_evaluation="nemo_literal").
                    eta_now=(
                        _nemo_ws_zdf_eta_kmm
                        if _nemo_ws_zdf_eta_kmm is not None
                        else state.eta.data),
                    u_now=state.u.data, v_now=state.v.data,
                    effective_K_test_override=_vertical_K_test_override,
                    nemo_aimp_tracer_w=_nemo_ws_aimp_tracer_w,
                    nemo_tracer_content_rhs=_nemo_ws_tracer_content_rhs,
                    nemo_aimp_momentum_w_u=_nemo_ws_aimp_momentum_w_u,
                    nemo_aimp_momentum_w_v=_nemo_ws_aimp_momentum_w_v,
                    z_coord=z_coord, config=config, iwm_fields=iwm_fields)
                if _return_live_stage_operands:
                    (state_new, tke_new,
                     _nemo_ws_live_tke_entry,
                     _nemo_ws_live_tke_statement_trace) = _tke_result
                else:
                    state_new, tke_new, _nemo_ws_vertical_solve_trace = (_tke_result if _return_vertical_solve_trace else (*_tke_result, None))
            else:
                _n2_tracers = self._n2_before_advection_tracers(state, z_coord=z_coord, config=config)
                _n2_tracers_before = self._n2_nemo_before_tracers(state, z_coord=z_coord, config=config)
                _zdf_result = self._apply_implicit_vertical_mixing(
                    state_new, dt, surface_forcing,
                    K_v_phys=tend.K_v, A_v_phys=tend.A_v,
                    K33_iso=k33_implicit, dt_mom=dt_mom,
                    surface_tracer_forcing=tend.surface_tracer_forcing,
                    tracer_source=tend.tracer_source,
                    grid=_grid, n2_tracers=_n2_tracers,
                    n2_tracers_before=_n2_tracers_before,
                    tke_n2_bundle=_tke_n2_bundle,
                    # NEMO e3w(Kmm) divisor (#1226 W1): see the sibling call.
                    eta_now=(
                        _nemo_ws_zdf_eta_kmm
                        if _nemo_ws_zdf_eta_kmm is not None
                        else state.eta.data),
                    u_now=state.u.data, v_now=state.v.data,
                    effective_K_test_override=_vertical_K_test_override,
                    nemo_aimp_tracer_w=_nemo_ws_aimp_tracer_w,
                    nemo_tracer_content_rhs=_nemo_ws_tracer_content_rhs,
                    nemo_aimp_momentum_w_u=_nemo_ws_aimp_momentum_w_u,
                    nemo_aimp_momentum_w_v=_nemo_ws_aimp_momentum_w_v, return_tracer_solve_trace=_return_vertical_solve_trace,
                    z_coord=z_coord, config=config, iwm_fields=iwm_fields)
                state_new, _nemo_ws_vertical_solve_trace = (_zdf_result if _return_vertical_solve_trace else (_zdf_result, None))
        if _return_tracer_process_trace:
            _nemo_ws_process_Taa = state_new.T.data
        if tke_new is not None:
            # Veros order (integrate_tke): the implicit solve writes
            # tke[taup1] FIRST, then the superbee-advection AB2 increment is
            # added to it (tke.py:315-323). Advects the carried state.tke
            # (tke[tau]) by the pre-step state.u/v (u[tau]); dt here is the
            # TRACER dt (Veros dt_tracer). Static gate ⇒ default adds no ops.
            if self._tke_advection_active():
                tke_new, _dtke_field = self._apply_tke_advection(
                    state, tke_new, dt, grid=_grid, z_coord=z_coord, config=config)
                state_new = state_new._replace(dtke=_dtke_field)
            state_new = state_new._replace(
                tke=Field(data=tke_new, name="tke",
                          dims=("lat", "lon", "level"), units="m^2/s^2"),
            )

        if self._nemo_ws_test_hooks.expose_stage3_raw_momentum:
            # WRITE-only capture at NEMO's raw-Kaa boundary: the implicit
            # dyn_zdf result is complete, while the depth-mean replacement
            # immediately below has not yet run (stprk3_stg.F90:430-448).
            _nemo_ws_exposed_stage3_raw = (
                state_new.u.data, state_new.v.data)

        if _ws_stage3_correction is not None:
            # NEMO stprk3_stg.F90:440,444-445, applied HERE because :437-446
            # runs after the CALL dyn_zdf at :430.  Same closure stages 1 and
            # 2 call, same target uu_b(Kaa), same e3u_0/hu_0 weights — one
            # correction per stage, exactly as NEMO writes it.  Sign
            # convention: an ADDITIVE column-uniform shift, so the baroclinic
            # deviation u′ is untouched and the depth integral becomes exactly
            # the barotropic transport.
            #
            # UNCONDITIONAL, and deliberately so: NEMO's banner at
            # stprk3_stg.F90:433 corrects every stage on every card, and this
            # is the stage-3 correction the ladder deferred, not the separate
            # ``nemo_stage_mean_imposition`` re-imposition below.  Gating it on
            # that flag would DELETE the correction on the two tank cards,
            # which resolve it False — measured at 8.4e-03 on OVERFLOW's u,
            # 8.7 per cent of the field.
            _stage3_corr, _stage3_tu, _stage3_tv = _ws_stage3_correction
            _u_after, _v_after = _stage3_corr(
                state_new.u.data, state_new.v.data, _stage3_tu, _stage3_tv)
            # The cyclic wrap column travelled WITH the correction before it
            # was deferred; re-apply it here so a periodic card keeps the
            # invariant the stage used to leave behind.  Inert on all four
            # closed-basin cards, where both columns are masked to zero.
            _u_after = _u_after.at[:, -1].set(_u_after[:, 0])
            state_new = state_new._replace(
                u=state_new.u.replace(data=_u_after),
                v=state_new.v.replace(data=_v_after),
            )
        elif _impose_mean:
            # NEMO stprk3_stg.F90:440: uu += (uu_b(Kaa) − Σ e3u_0·uu·r1_hu_0)
            # ·umask — the 3D velocity's depth mean is REPLACED by the
            # barotropic solution after the implicit solve, uniformly over
            # the column. Sign convention: an ADDITIVE column-uniform shift,
            # so the baroclinic deviation u′ is untouched (budget: the
            # depth-integral becomes exactly the barotropic transport).
            _u_mean_now, _v_mean_now = self._fixed_depth_means(state_new, z_coord=z_coord, config=config, grid=grid)
            _du = (_u_mean_baro - _u_mean_now)[..., jnp.newaxis]
            _dv = (_v_mean_baro - _v_mean_now)[..., jnp.newaxis]
            _um = state.u_mask.data[..., jnp.newaxis]
            _vm = state.v_mask.data[..., jnp.newaxis]
            state_new = state_new._replace(
                u=state_new.u.replace(data=state_new.u.data + _du * _um),
                v=state_new.v.replace(data=state_new.v.data + _dv * _vm),
            )

        # 9. Conservation fixers
        if _cfg_b.use_conservation_fixer and _apply_implicit_vmix:
            state_new = ocean_conservation_fixer(
                state_new, state, _grid, _zc, _cfg_b,
            )

        # Prescribed-flow lever: FINAL re-pin.  The implicit vertical-mixing
        # solve above mixes MOMENTUM too (and ab2_scope="advective" adds a
        # weight-1.0 momentum dissipation increment; the volume fixer can
        # shift eta): those solves RUN on the pinned flow — their T/S/TKE
        # couplings (TKE shear production, realized K_diss_v) therefore see
        # the prescribed circulation, which is the intended physics — but
        # their momentum/eta result is DISCARDED here (cheapest correct
        # route: run-and-discard, no surgically skipped sub-solves).
        if _pflow is not None:
            state_new = state_new._replace(
                u=state_new.u.replace(data=_u_pin),
                v=state_new.v.replace(data=_v_pin),
                eta=state_new.eta.replace(data=_eta_pin),
            )

        # Private oracle-fidelity seam: NEMO's stage dumps are written inside
        # stprk3_stg immediately after each instantaneous Kaa update.  Expose
        # stages 1/2 only after the ordinary step has completed so the
        # diagnostic hook cannot perturb later stages.  Stage 3 is the normal
        # returned prognostic velocity and therefore needs no substitution.
        if _nemo_ws_exposed_momentum_stage is not None:
            _stage_u, _stage_v = _nemo_ws_exposed_momentum_stage
            state_new = state_new._replace(
                u=state_new.u.replace(data=_stage_u),
                v=state_new.v.replace(data=_stage_v),
            )
        if _nemo_ws_exposed_stage2_raw is not None:
            _raw_u, _raw_v = _nemo_ws_exposed_stage2_raw
            state_new = state_new._replace(
                u=state_new.u.replace(data=_raw_u),
                v=state_new.v.replace(data=_raw_v),
            )
        if _nemo_ws_exposed_tracer_stage is not None:
            # The stage T/S/eta exactly as handed to the stage eos+dyn_hpg.
            _stage_T, _stage_S, _stage_eta = _nemo_ws_exposed_tracer_stage
            state_new = state_new._replace(
                T=state_new.T.replace(data=_stage_T),
                S=state_new.S.replace(data=_stage_S),
                eta=state_new.eta.replace(data=_stage_eta),
            )
        if _nemo_ws_exposed_tracer_boundary is not None:
            # Substitute only after the production-JIT step has completed.
            # The exposed arrays are source-order ts(Krhs), not prognostics.
            _rhs_T, _rhs_S = _nemo_ws_exposed_tracer_boundary
            state_new = state_new._replace(
                T=state_new.T.replace(data=_rhs_T),
                S=state_new.S.replace(data=_rhs_S),
            )
        if _nemo_ws_exposed_tracer_transport is not None:
            _zfu, _zfv, _zfw = _nemo_ws_exposed_tracer_transport
            state_new = state_new._replace(
                u=state_new.u.replace(data=_zfu),
                v=state_new.v.replace(data=_zfv),
                T=state_new.T.replace(data=_zfw[..., :state_new.T.data.shape[-1]]),
            )
        if _nemo_ws_exposed_stage1_wzv is not None:
            # Diagnostic substitution happens only after the compiled step and
            # all of its ordinary consumers have completed.
            state_new = state_new._replace(
                T=state_new.T.replace(
                    data=_nemo_ws_exposed_stage1_wzv[
                        ..., :state_new.T.data.shape[-1]
                    ]
                )
            )
        if _nemo_ws_exposed_stage_face_r3 is not None:
            # WRITE-only: the ordinary step has already completed and nothing
            # downstream reads these slots.
            _r3u, _r3v = _nemo_ws_exposed_stage_face_r3
            state_new = state_new._replace(
                u=state_new.u.replace(
                    data=jnp.broadcast_to(
                        _r3u[..., None], state_new.u.data.shape)),
                v=state_new.v.replace(
                    data=jnp.broadcast_to(
                        _r3v[..., None], state_new.v.data.shape)),
            )
        if _nemo_ws_exposed_stage1_transport_operand is not None:
            _operand_u, _operand_v = _nemo_ws_exposed_stage1_transport_operand
            state_new = state_new._replace(
                u=state_new.u.replace(data=_operand_u),
                v=state_new.v.replace(data=_operand_v),
            )
        if _nemo_ws_exposed_momentum_operator is not None:
            _op_u, _op_v = _nemo_ws_exposed_momentum_operator
            state_new = state_new._replace(
                u=state_new.u.replace(data=_op_u),
                v=state_new.v.replace(data=_op_v),
            )
        if _nemo_ws_exposed_stage1_rhs is not None:
            _rhs_u, _rhs_v = _nemo_ws_exposed_stage1_rhs
            state_new = state_new._replace(
                u=state_new.u.replace(data=_rhs_u),
                v=state_new.v.replace(data=_rhs_v),
            )
        if _nemo_ws_exposed_stage1_raw is not None:
            _raw_u, _raw_v = _nemo_ws_exposed_stage1_raw
            state_new = state_new._replace(
                u=state_new.u.replace(data=_raw_u),
                v=state_new.v.replace(data=_raw_v),
            )
        if _nemo_ws_exposed_stage2_rhs is not None:
            _rhs_u, _rhs_v = _nemo_ws_exposed_stage2_rhs
            state_new = state_new._replace(
                u=state_new.u.replace(data=_rhs_u),
                v=state_new.v.replace(data=_rhs_v),
            )
        if _nemo_ws_exposed_stage3_rhs is not None:
            _rhs_u, _rhs_v = _nemo_ws_exposed_stage3_rhs
            state_new = state_new._replace(
                u=state_new.u.replace(data=_rhs_u),
                v=state_new.v.replace(data=_rhs_v),
            )
        if _nemo_ws_exposed_stage3_raw is not None:
            _raw_u, _raw_v = _nemo_ws_exposed_stage3_raw
            state_new = state_new._replace(
                u=state_new.u.replace(data=_raw_u),
                v=state_new.v.replace(data=_raw_v),
            )
        if _nemo_ws_pre_implicit_state is not None:
            # Substitute only after the production step and its conservation
            # checks have completed, preserving a WRITE-only diagnostic.
            state_new = _nemo_ws_pre_implicit_state

        # NEMO's after-SSH slot is written HERE, not in the compiled step
        # wrapper, because the wrapper is not the only way in: the OMIP scan
        # driver, the SPMD lane and the OMIP2 applicator all call _step_impl
        # directly, and the READ (in the wzv call below the tendency) fires on
        # every one of them.  A write that reached fewer callers than the read
        # would hand those drivers a frozen slot with no error.  Before the
        # storage cast, so the slot is cast with every other leaf and a scan
        # carry's dtypes stay stable.
        state_new = self._carry_nemo_rk3_after_ssh(state, state_new)
        state_new = cast_pytree(state_new, None, "storage", allow_downcast=True)
        if _return_ldf_diagnostic_trace:
            if _nemo_ws_ldf_diagnostics is None:
                raise ValueError("WS-RK3 LDF diagnostic trace is incomplete")
            return _NEMOWSLdfDiagnosticTrace(
                state_after=state_new,
                ldf_diagnostics=_nemo_ws_ldf_diagnostics)
        if _return_tracer_process_trace:
            if (_nemo_ws_process_qco is None
                    or _nemo_ws_process_boundaries is None
                    or _nemo_ws_process_Taa is None
                    or _nemo_ws_qsr_association is None):
                raise ValueError("WS-RK3 tracer process trace is incomplete")
            if (_return_vertical_solve_trace
                    and _nemo_ws_vertical_solve_trace is None):
                raise ValueError("WS-RK3 vertical solve trace is incomplete")
            _qbb, _qmm, _qaa, _, _, _ = _nemo_ws_process_qco
            return _NEMOWSTracerProcessTrace(
                state_after=state_new,
                Tbb=state.T.data,
                q_Kbb=_qbb,
                q_Kmm=_qmm,
                q_Kaa=_qaa,
                boundaries=_nemo_ws_process_boundaries,
                Taa=_nemo_ws_process_Taa,
                qsr_association=_nemo_ws_qsr_association,
                vertical_solve=_nemo_ws_vertical_solve_trace,
                fct_activity=_nemo_ws_fct_activity, ldf_diagnostics=_nemo_ws_ldf_diagnostics,
            )
        if _return_live_stage_operands:
            if (getattr(_cfg_b, "momentum_time_integrator", "euler")
                    != "rk3_ws"):
                raise ValueError(
                    "live stage operands require momentum_time_integrator="
                    "'rk3_ws'")
            if (any(value is None for value in _nemo_ws_live_operator_operands)
                    or _nemo_ws_live_stage_states is None
                    or _nemo_ws_live_stage_geometry is None or _nemo_ws_live_stage_raw is None or _nemo_ws_live_stage_rhs is None or _nemo_ws_live_baro_geometry is None  # noqa: E501
                    or _nemo_ws_live_stage_qco is None
                    or _nemo_ws_live_stage1_full_rhs is None
                    or _nemo_ws_live_stage1_rhs_walk is None
                    or _nemo_ws_live_tke_entry is None
                    or _nemo_ws_live_tke_statement_trace is None
                    or _nemo_ws_live_slow_forcing_producer is None):
                raise ValueError("live WS-RK3 operand trace is incomplete")
            return _NEMOWSLiveOperandTrace(
                state_after=state_new,
                stage_states=_nemo_ws_live_stage_states,
                operator_operands=tuple(_nemo_ws_live_operator_operands),
                stage_geometry=_nemo_ws_live_stage_geometry,
                stage_qco=_nemo_ws_live_stage_qco,
                stage_coefficients=(
                    (dt / 3.0, 1.0 / (dt / 3.0)),
                    (dt / 2.0, 1.0 / (dt / 2.0)),
                    (dt, 1.0 / dt),
                ),
                tke_entry=_nemo_ws_live_tke_entry,
                tke_statement_trace=_nemo_ws_live_tke_statement_trace,
                barotropic_targets=(target_u, target_v, Hu_avg, Hv_avg,
                                    state_new.eta.data,
                                    (_eta_live_one_third,
                                     _eta_live_one_half,
                                     state_new.eta.data)),
                slow_forcing_producer=(
                    _nemo_ws_live_slow_forcing_producer),
                stage_rhs=_nemo_ws_live_stage_rhs,
                stage1_full_rhs=_nemo_ws_live_stage1_full_rhs,
                stage1_rhs_walk=_nemo_ws_live_stage1_rhs_walk,
                stage_raw_velocities=_nemo_ws_live_stage_raw,
                barotropic_correction_geometry=_nemo_ws_live_baro_geometry,
                stage_outputs=(
                    (u1_corr, v1_corr, _T_stage1, _S_stage1,
                     _eta_live_one_third),
                    (u2_corr, v2_corr, _T_stage2, _S_stage2,
                     _eta_live_one_half),
                    (state_new.u.data, state_new.v.data,
                     state_new.T.data, state_new.S.data,
                     state_new.eta.data),
                ),
                # WRITE-only: the per-stage tracer source rates the stage
                # helper consumes, each with the live top-cell thickness that
                # stage divided by, so a gate can bind on the PRODUCTION
                # statement instead of re-implementing it.
                stage_tracer_sources=(
                    _stage_source_rates,
                    (h_k_old, _h_live_one_third, _h_live_one_half),
                ),
            )
        if not _apply_implicit_vmix:
            # Faithful AB2 path: return the explicit-only state plus the
            # implicit-mixing diffusivity profiles (evaluated from u^n, like
            # Veros's du_mix / kappaH) and the WITHHELD surface-tracer forcing.
            # ``_ab2_step`` applies implicit vertical mixing ONCE after the AB2
            # extrapolation — adding the surface forcing INSIDE that implicit
            # solve (so it is NOT AB2-extrapolated) — and runs the conservation
            # fixer once on the final state.  ``tend.surface_tracer_forcing`` is
            # ``None`` unless ``surface_forcing_implicit`` is on ⇒ bit-identical.
            # The prognostic-TKE source (eke_diss_iw from THIS step's EKE update
            # on state_new + K_diss_bot from the tendency) is assembled here so
            # ``_ab2_step``'s single implicit-mixing call can advance state.tke
            # exactly once per step (None unless prognostic TKE is active ⇒
            # bit-identical).
            _tke_src = (self._assemble_tke_source(state, state_new, tend, z_coord=z_coord, config=config)
                        if self._tke_prognostic_active() else None)
            # AB2 "advective" scope: the weight-1.0 dissipative INCREMENT
            # (momentum lateral friction + bottom drag; tracer lateral
            # diffusion + GM/Redi) — NOT applied to state_new here; ``_ab2_step``
            # adds it after the AB2 extrapolation. ``None`` under "total".
            _diss_incr = (None if _diss_dT_incr is None else
                          (_diss_dT_incr, _diss_dS_incr,
                           _diss_du_incr, _diss_dv_incr))
            # NB: ``tracer_source`` (EXT-N2) is APPENDED so the positional
            # contract of the first six slots is unchanged (consumed by
            # _ab2_step and tests/ocean/unit/test_ab2_scope.py).
            _aux = (tend.K_v, tend.A_v, k33_implicit,
                    tend.surface_tracer_forcing, _tke_src,
                    _diss_incr, tend.tracer_source)
            if _return_raw_kaa_qco:
                # Transient within-step operand, deliberately outside the
                # prognostic/restart state tree: NEMO's Kaa SSH immediately
                # after dyn_spg_ts and before legoESM's global eta projection.
                _aux = _aux + (
                    _eta_after_spg_literal, Hu_avg, Hv_avg)
            return state_new, _aux
        return state_new

    def _fixed_depth_means(self, st, z_coord=None, config=None, grid=None):
        """Thickness-weighted depth means of u, v on FIXED reference
        thicknesses (NEMO ``e3u_0``/``r1_hu_0``, the linssh convention used by
        the stprk3_stg:440 zub correction). Face thicknesses by the min-rule,
        matching the barotropic solver's depth average
        (``_depth_average_to_faces``) so imposition restores exactly the mean
        the barotropic solve set."""
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        _grid = self.grid if grid is None else grid  # SPMD band override
        from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
            barotropic_depth_average_to_faces as _depth_average_to_faces,
        )
        from legoesm.ocean.vertical import compute_layer_thickness
        h_k = compute_layer_thickness(
            jnp.zeros_like(st.eta.data), st.H_bathy.data, _zc,
            min_water_column_m=_cfg_b.min_water_column_m,
        ).astype(st.u.data.dtype)
        return _depth_average_to_faces(
            st.u.data, st.v.data, h_k,
            jnp.asarray(_cfg_b.min_water_column_m, dtype=st.u.data.dtype),
            st.land_mask.data, st.u_mask.data, st.v_mask.data, _grid)

    def _tke_prognostic_active(self) -> bool:
        """True iff a prognostic-TKE-carrying vertical-mixing closure is active.

        Static Python predicate (config-only, no traced values): either the
        Gaspar/Burchard ``"tke"`` scheme with ``tke.prognostic=True``, OR the
        ``"catke"`` scheme (CATKE is ALWAYS prognostic — Wagner et al. 2025).
        Both carry ``OceanState.tke`` and run one backward-Euler TKE step per
        model step.  Gates the prognostic-TKE carry in the model step (default
        False ⇒ the Mode-B diagnostic chain).
        """
        physics = getattr(self.config, "physics", None)
        if physics is None:
            return False
        vmix = getattr(physics, "vertical_mixing", None)
        if vmix is None:
            return False
        if vmix.scheme == "catke":
            return True
        if vmix.scheme != "tke":
            return False
        return bool(getattr(vmix.tke, "prognostic", False))

    def _tke_post_mixing_active(self) -> bool:
        """True iff the prognostic TKE runs the Veros POST-MIXING step order.

        Static Python predicate (config-only): prognostic TKE is on AND
        ``vertical_mixing.tke.buoyancy_timing == "post_mixing_veros"``. The
        K profiles for the tracer/momentum solves then come from the CARRIED
        TKE (Veros set_tke_diffusivities from tke[tau]) and the TKE budget is
        solved AFTER the implicit tracer mixing, charging the POST-mixing N²
        plus the surface buoyancy-flux P_diss_v slot (Veros
        thermodynamics.py:385-388 + tke.py:142). Default False ⇒ the
        pre-mixing legacy ordering, bit-identical.
        """
        if not self._tke_prognostic_active():
            return False
        # Veros post-mixing ordering is a TKE-scheme feature (not CATKE).
        if self.config.physics.vertical_mixing.scheme != "tke":
            return False
        tke_cfg = self.config.physics.vertical_mixing.tke
        return (getattr(tke_cfg, "buoyancy_timing", "pre_mixing")
                == "post_mixing_veros")

    def _n2_before_advection_tracers(self, entry_state, z_coord=None, config=None):
        """Before-advection (Nnow) T/S for the vmix diffusivity-stage N².

        Static Python predicate (config-only): returns ``(T, S)`` from the
        STEP-ENTRY state when ``vertical_mixing.scheme=="tke"`` and
        ``tke.n2_before_advection`` is set — the NEMO ``eosbn2`` sequencing
        (``bn2(Nnow)`` at step start, before ``fct2`` tracer advection drifts
        the deepest wet cell). ``None`` (default) ⇒ the closure keeps sampling
        N² on the post-advection state ⇒ BIT-IDENTICAL.

        Both the ``adiabatic`` (Veros parcel-displacement) and ``nemo_bn2``
        (NEMO eosbn2 S-EOS) N² paths read the T/S contrast — the
        ``nemo_dino_kamm`` card's actual N² source (see
        ``compute_N2``/``_shared.py``); any OTHER ``n2_mode`` with the flag
        set would be a SILENT no-op, so raise (dispatch hardening — a
        mis-wired flag must fail loudly).
        """
        # z_coord accepted for uniform SPMD forwarding; this helper reads no vertical geometry directly.
        _cfg_b = self.config if config is None else config  # SPMD band override
        vmix = getattr(getattr(_cfg_b, "physics", None),
                       "vertical_mixing", None)
        if vmix is None or vmix.scheme != "tke":
            return None
        if not getattr(vmix.tke, "n2_before_advection", False):
            return None
        if getattr(vmix.tke, "n2_mode", "insitu") not in (
                "adiabatic", "nemo_bn2"):
            raise ValueError(
                "vertical_mixing.tke.n2_before_advection=True requires "
                "n2_mode='adiabatic' or 'nemo_bn2' (the only N² paths that "
                f"read the T/S contrast); got n2_mode={vmix.tke.n2_mode!r}.")
        return (entry_state.T.data, entry_state.S.data)

    def _n2_nemo_before_tracers(self, entry_state, z_coord=None, config=None):
        """TRUE leap-frog BEFORE (Nbb) T/S for the rn2b consumers (T8/T13).

        Static Python predicate: returns ``(T_before, S_before)`` — the
        carried leap-frog before-state, one FULL step behind
        ``entry_state`` — when ``vertical_mixing.tke.tke_n2_time_level``
        is "nemo_before"; ``None`` (default) ⇒ BIT-IDENTICAL (rn2b
        consumers reuse the same N² as rn2, unchanged).

        ``entry_state`` MUST be the step-entry state (before rebinding to
        ``state_new``), matching ``_n2_before_advection_tracers``'s
        convention. Construction guarantees ``outer_integrator`` in
        ``("leapfrog", "nemo_mlf")`` (the fields EXIST as NamedTuple slots,
        populated identically by either step method's Asselin-filter tail;
        P2) but NOT that they are
        POPULATED on every possible caller: ``_leapfrog_step``'s Euler-start
        branch (#1317 fix) now seeds a LOCAL before:=now copy before its
        first ``_step_impl`` call — matching NEMO's own cold-start
        convention (``istate.F90:97-99``/``135-137``: ``ts(:,:,:,:,Kmm) =
        ts(:,:,:,:,Kbb)`` before ``stp_MLF`` is ever entered, so Nbb==Nnn
        identically on the first step; ``stpmlf.F90:114-117``
        ``l_1st_euler -> rDt=rn_Dt`` then makes the leap-frog combine
        degenerate exactly to forward-Euler) — so a fresh/from-rest state
        never reaches this predicate with ``T_before=None``. A state that
        DOES still reach here with ``T_before=None`` is a genuinely
        mis-wired caller (e.g. ``_step_impl`` invoked directly, bypassing
        ``_leapfrog_step``, on a hand-built state that was never seeded or
        bridged) — raise loudly rather than crash on ``NoneType.data``
        (AttributeError) or silently fall back to ``entry_state.T``/``.S``
        (which would mask that mis-wiring).
        """
        # z_coord accepted for uniform SPMD forwarding; this helper reads no vertical geometry directly.
        _cfg_b = self.config if config is None else config  # SPMD band override
        vmix = getattr(getattr(_cfg_b, "physics", None),
                       "vertical_mixing", None)
        if vmix is None or vmix.scheme != "tke":
            return None
        _conv = getattr(getattr(_cfg_b, "physics", None), "convection", None)
        _evd = (getattr(_conv, "enhanced_diffusion", None)
                if getattr(_conv, "scheme", "none") == "enhanced_diffusion"
                else None)
        if getattr(_cfg_b, "momentum_time_integrator", "euler") == "rk3_ws":
            # NEMO RK3 has no carried leap-frog BEFORE array.  Its whole-step
            # entry is Nbb for BOTH independent consumers: TKE when
            # tke_n2_time_level=nemo_before, and EVD when its two-level trigger
            # asks for rn2b (stprk3.F90:154-165).  Do not tie this mapping to
            # EVD's selector: the one-variable EVD ablation must not make the
            # still-live TKE consumer invent state.T_before/S_before.
            _tke_uses_entry_as_nbb = (
                getattr(vmix.tke, "tke_n2_time_level", "step_entry")
                == "nemo_before")
            _evd_uses_entry_as_nbb = (
                _evd is not None
                and getattr(_evd, "evd_n2_time_level", "solver_state")
                == "nemo_now_before")
            if _tke_uses_entry_as_nbb or _evd_uses_entry_as_nbb:
                return (entry_state.T.data, entry_state.S.data)
        if getattr(vmix.tke, "tke_n2_time_level", "step_entry") != "nemo_before":
            return None
        if entry_state.T_before is None or entry_state.S_before is None:
            raise ValueError(
                'vertical_mixing.tke.tke_n2_time_level="nemo_before" requires '
                "before-level tracers (state.T_before/S_before) to be "
                "populated, but they are None. A from-rest run through "
                "model.step()/_leapfrog_step already seeds before:=now on "
                "the Euler-start step (NEMO istate.F90 Kmm:=Kbb convention); "
                "a bridged/twin state must seed the leap-frog before-level "
                "fields before stepping — see kamm_twin_90d.py's "
                "--bridge-before (bridges NEMO's restart tb/sb/ub/vb onto "
                "state.{T,S,u,v}_before). This error firing means "
                "_step_impl was called directly on a state that was never "
                "seeded/bridged -- the caller must go through "
                "_leapfrog_step/step() or bridge the before-level state "
                "itself, not silently fall back to entry_state.T/.S."
            )
        return (entry_state.T_before.data, entry_state.S_before.data)

    def _tke_bottom_dirichlet(self, state, z_coord=None, config=None, grid=None):
        """NEMO bottom TKE BC value (T15; zdftke.F90:279-288), or None.

        Static Python predicate: returns the Dirichlet TKE value when
        ``vertical_mixing.scheme=="tke"`` and ``tke.bottom_tke_bc`` is set;
        ``None`` (default) ⇒ BIT-IDENTICAL (no bottom BC).

        Reuses the SAME NEMO bottom-drag rate + partial-cell bottom-level
        machinery as ``zdf_drag_in_matrix``
        (:func:`nemo_bottom_drag_rate_faces`'s T-point building blocks,
        :func:`nemo_effective_bottom_drag_r`) — single-owner doctrine, no
        re-derived drag coefficient.

        Takes the RAW face-staggered ``state`` (never the caller's collapsed
        ``cc_state``): zdftke's wet-only face masking cannot be reconstructed
        once u/v have been averaged to the T-point. The plain T-point average
        the drag rate needs is re-formed here with the SAME expression the
        caller uses, so the drag path stays bit-identical.
        """
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        _grid = self.grid if grid is None else grid  # SPMD band override
        vmix = getattr(getattr(_cfg_b, "physics", None),
                       "vertical_mixing", None)
        if vmix is None or vmix.scheme != "tke":
            return None
        if not getattr(vmix.tke, "bottom_tke_bc", False):
            return None
        if not isinstance(_zc, OceanPartialCellCoordinate):
            raise ValueError(
                "vertical_mixing.tke.bottom_tke_bc=True requires a "
                "partial-cell z-coordinate (bottom_level) — the flat-bottom "
                "case is not covered.")
        from legoesm import constants
        from legoesm.ocean.dynamics.ocean_tendency_common import (
            nemo_effective_bottom_drag_r, validate_bottom_drag_scheme,
        )
        _scheme = validate_bottom_drag_scheme(
            str(getattr(_cfg_b.bottom_drag, "bottom_drag_scheme",
                        "legacy")))
        if _scheme == "legacy":
            raise ValueError(
                "vertical_mixing.tke.bottom_tke_bc=True requires a NEMO "
                "bottom_drag_scheme ('nemo_quadratic' or 'nemo_loglayer'), "
                f"got 'legacy'.")
        h_k = _zc.h_partial
        _bl = jnp.maximum(_zc.bottom_level, 0)
        _bl_idx = _bl[..., jnp.newaxis]
        u_f, v_f = state.u.data, state.v.data
        if (u_f.shape[1] != h_k.shape[1] + 1
                or v_f.shape[0] != h_k.shape[0] + 1):
            raise ValueError(
                "_tke_bottom_dirichlet needs the RAW face-staggered state "
                "(u (n_lat, n_lon+1, nlev), v (n_lat+1, n_lon, nlev)) — the "
                "wet-only face masking below cannot be reconstructed from an "
                f"already-collapsed T-point field. Got u{u_f.shape} "
                f"v{v_f.shape} against T-shape {h_k.shape}.")
        # zdfdrg.F90:174-181 forms zut = uu(ji)+uu(ji-1) and then
        # SQRT(0.25*(zut^2+zvt^2)+ke0) — i.e. the PLAIN T-point average, with
        # no wet-only masking of its own (NEMO's uu is already umask'ed at
        # every level, dynzdf.F90:121-150).
        u_cc = 0.5 * (u_f[:, :-1, :] + u_f[:, 1:, :])
        v_cc = 0.5 * (v_f[:-1, :, :] + v_f[1:, :, :])
        u_bot = jnp.take_along_axis(u_cc, _bl_idx, axis=-1)[..., 0]
        v_bot = jnp.take_along_axis(v_cc, _bl_idx, axis=-1)[..., 0]
        h_bot = jnp.take_along_axis(h_k, _bl_idx, axis=-1)[..., 0]
        r_t = nemo_effective_bottom_drag_r(
            u_bot, v_bot, h_bot,
            scheme=_scheme,
            cd0=float(_cfg_b.bottom_drag.bottom_drag_cd0),
            cd_max=float(_cfg_b.bottom_drag.bottom_drag_cdmax),
            z0=float(_cfg_b.bottom_drag.bottom_drag_z0),
            ke0=float(_cfg_b.bottom_drag.bottom_drag_ke0),
            uc0=float(_cfg_b.bottom_drag.bottom_drag_uc0),
            von_karman=constants.kappa_von_karman,
        )
        from legoesm.ocean.physics.vertical_mixing.tke import (
            nemo_bottom_tke_dirichlet,
        )
        # zdftke's velocity convention is NOT zdfdrg's.  The drag rate above
        # takes the plain T-point average; the TKE bottom BC instead uses the
        # WET-ONLY SUM  zmsku*( uu(ji) + uu(ji-1) )  with
        # zmsku = 2 - umask(ji-1,jj,mbkt)*umask(ji,jj,mbkt)  and NO 0.5
        # (zdftke.F90:282-287; contrast zdfgls.F90:196-197, which writes the
        # same mask expression WITH the 0.5).  The missing 0.5 is structural,
        # not a NEMO slip: it cancels the 0.5 already inside the 0.001875
        # prefactor (= (rn_ebb0/rho0)*0.5, zdftke.F90:284), leaving
        # en_bot = (rn_ebb0/rho0)*Cd|U|^2 ∝ u_*^2 — the same form as the
        # surface BC en(1) = zbbrau*taum (:266).
        #
        # The faces are masked EXPLICITLY here.  NEMO's uu is umask'ed at
        # every level (dynzdf.F90:121-150) so its bare sum is already
        # wet-only; legoESM's prognostic u carries only the 2-D column mask,
        # and the barotropic correction adds a uniform-in-k increment, so a
        # sub-seafloor face holds a small NON-ZERO velocity.  Reusing the
        # canonical `compute_face_masks_3d` (rather than re-deriving the
        # geometry) also inherits its seam-wall and meridional-periodicity
        # exclusions, which a hand-rolled tmask product would silently drop.
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            compute_face_masks_3d,
        )
        um3, vm3 = compute_face_masks_3d(_zc.is_active, _grid)
        um3 = um3.astype(h_k.dtype)
        vm3 = vm3.astype(h_k.dtype)

        def _at_bottom(a):
            return jnp.take_along_axis(a, _bl_idx, axis=-1)[..., 0]

        # u-face i is the WEST face of T-cell i, face i+1 the EAST face
        # (latlon_cgrid_operators.py:3573) — NEMO's umask(ji-1) / umask(ji).
        m_uw, m_ue = _at_bottom(um3[:, :-1, :]), _at_bottom(um3[:, 1:, :])
        m_vs, m_vn = _at_bottom(vm3[:-1, :, :]), _at_bottom(vm3[1:, :, :])
        u_sum = (2.0 - m_uw * m_ue) * (
            _at_bottom(u_f[:, :-1, :]) * m_uw + _at_bottom(u_f[:, 1:, :]) * m_ue)
        v_sum = (2.0 - m_vs * m_vn) * (
            _at_bottom(v_f[:-1, :, :]) * m_vs + _at_bottom(v_f[1:, :, :]) * m_vn)
        # NEMO closes the line with  * ssmask(ji,jj)  (zdftke.F90:288): a dry
        # column gets 0, not rn_emin.  `_bl` is clamped to >= 0 there, so
        # without this the reduction's "centre cell wet at its own mbkt"
        # premise would not hold on land either.
        ssmask = _at_bottom(_zc.is_active.astype(h_k.dtype))
        return ssmask * nemo_bottom_tke_dirichlet(r_t, u_sum, v_sum, vmix.tke)

    def _tke_bottom_level(self, z_coord=None, config=None):
        """Per-column T-point bottom-cell index for TKE consumers, or None.

        The bottom Dirichlet placement and NEMO's literal Langmuir crossing
        fallback independently consume ``mbkt``. Route the coordinate-owned
        value when either is active. Other configurations retain ``None`` so
        the silent-unused-operand guard remains effective.
        """
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        vmix = getattr(getattr(_cfg_b, "physics", None),
                       "vertical_mixing", None)
        if vmix is None or vmix.scheme != "tke":
            return None
        needs_bottom = bool(getattr(vmix.tke, "bottom_tke_bc", False))
        needs_langmuir = (
            bool(getattr(vmix.tke, "lc", False))
            and getattr(vmix.tke, "tke_langmuir_evaluation", "vectorized")
            == "nemo_literal")
        if not (needs_bottom or needs_langmuir):
            return None
        if not isinstance(_zc, OceanPartialCellCoordinate):
            return None
        return _zc.bottom_level

    def _seed_tke_preclosure_carry(self, state):
        """Seed NEMO's cold-start ``avm_k/avt_k`` coefficient memory.

        This helper is intentionally shared by direct :meth:`step` and
        :meth:`seed_scan_carry`: restart bridges arrive with the fields
        populated and are left untouched, while a true cold start reproduces
        ``zdf_phy_init``'s background-times-``wmask`` construction.  Interior
        legoESM W index ``k`` is NEMO W level ``k+2``, so the exact partial-
        depth mask is ``is_active[..., 1:]``.
        """
        if not self._tke_prognostic_active():
            return state
        tke_cfg = self.config.physics.vertical_mixing.tke
        if (getattr(tke_cfg, "tke_preclosure_coeff_source",
                    "current_subiteration") != "carried_previous_step"):
            return state
        literal_matrix = (getattr(tke_cfg, "tke_matrix_evaluation", "factored")
                          == "nemo_literal")
        # The surface avm_k is produced by the closure ONLY under the nemo_z0
        # face assembly (tke.py: _K_M_surface stays None otherwise, and the
        # carried path demands preclosure_K_M_surface only in that same case).
        # Seeding it unconditionally made every step-2 state PARTIAL: the
        # post-solve writeback correctly stores None for it under
        # "interior_pinned", so the guard below then rejected the state the
        # model had just produced. That made carried_previous_step unusable on
        # any card except the nemo_z0 ones. Required set now matches what the
        # closure actually emits, exactly as tke_dissl already did.
        # codex 9693003 [HIGH]: nemo_z0 is necessary but NOT sufficient. The
        # closure also needs the ln_mxl0 anchor, which _mxl0_surface_anchor
        # returns only for tke_mxl_choice 3 or 4 (tke.py:176-177), so
        # nemo_z0 + choice 1/2 writes None and hit the same partial-state
        # crash this guard was meant to cure.
        surface_z0 = (
            getattr(tke_cfg, "tke_surface_bc_level",
                    "interior_pinned") == "nemo_z0"
            and int(getattr(tke_cfg, "tke_mxl_choice", 0)) in (3, 4))
        carry_fields = (state.tke_avm, state.tke_avt) + (
            (state.tke_avm_surface,) if surface_z0 else ()) + (
                (getattr(state, "tke_dissl", None),)
                if literal_matrix else ())
        n_present = sum(field is not None for field in carry_fields)
        if n_present == len(carry_fields):
            return state
        if n_present:
            raise ValueError(
                "carried_previous_step coefficient memory is partially "
                "populated: tke_avm, tke_avt, the nemo_z0 tke_avm_surface and "
                "any literal-matrix tke_dissl must be "
                "all present for a restart/continued state or all None for "
                "a true cold start")

        from legoesm.core.field import Field
        lm = state.land_mask.data
        nlev = state.T.data.shape[-1]
        dtype = state.T.data.dtype
        is_active = getattr(self.z_coord, "is_active", None)
        if is_active is None:
            wet_w = ((lm[..., None] > 0.5)
                     * jnp.ones((1, 1, nlev - 1), dtype=bool))
        else:
            wet_w = jnp.asarray(is_active[..., 1:], dtype=bool)
            if wet_w.shape != lm.shape + (nlev - 1,):
                raise ValueError(
                    "partial-cell W mask shape does not match TKE carry: "
                    f"got {wet_w.shape}, expected {lm.shape + (nlev - 1,)}")
            wet_w = wet_w & (lm[..., None] > 0.5)

        if state.tke_avm is None:
            avm0 = (jnp.ones((1, 1, nlev - 1), dtype=dtype)
                    * jnp.asarray(tke_cfg.kappaM_min, dtype=dtype))
            state = state._replace(tke_avm=Field(
                data=jnp.where(wet_w, avm0, 0.0), name="tke_avm",
                dims=("lat", "lon", "level"), units="m^2/s"))
        if state.tke_avt is None:
            avt0 = (jnp.ones((1, 1, nlev - 1), dtype=dtype)
                    * jnp.asarray(tke_cfg.kappaH_min, dtype=dtype))
            state = state._replace(tke_avt=Field(
                data=jnp.where(wet_w, avt0, 0.0), name="tke_avt",
                dims=("lat", "lon", "level"), units="m^2/s"))
        if surface_z0 and state.tke_avm_surface is None:
            avms0 = (jnp.asarray(tke_cfg.kappaM_min, dtype=dtype)
                     * lm.astype(dtype))
            state = state._replace(tke_avm_surface=Field(
                data=avms0, name="tke_avm_surface",
                dims=("lat", "lon"), units="m^2/s"))
        if literal_matrix and getattr(state, "tke_dissl", None) is None:
            # zdf_phy_alloc_init cold-start value before the first tke_avn.
            dissl0 = jnp.asarray(1.0e-12, dtype=dtype)  # coeff-ok: NEMO init
            state = state._replace(tke_dissl=Field(
                data=jnp.where(wet_w, dissl0, 0.0), name="tke_dissl",
                dims=("lat", "lon", "level"), units="s^-1"))
        return state

    def _tke_step_entry_n2_bundle(
        self, state, *, z_coord=None, config=None,
    ):
        """Build NEMO's pre-``zdf_phy`` rn2/rn2b/live-geometry bundle."""
        _zc = self.z_coord if z_coord is None else z_coord
        _cfg_b = self.config if config is None else config
        vmix = getattr(getattr(_cfg_b, "physics", None),
                       "vertical_mixing", None)
        _tke_active = vmix is not None and vmix.scheme == "tke"
        _gm = getattr(_cfg_b, "gm_redi", None)
        _gm_carried = (_gm is not None and getattr(
            _gm, "slope_n2_evaluation", "recompute") == "carried_step_entry")
        if _tke_active:
            tke_cfg = vmix.tke
            stage = getattr(
                tke_cfg, "tke_n2_evaluation_stage", "implicit_solve_state")
            if stage == "implicit_solve_state" and not _gm_carried:
                return None
            if stage not in ("implicit_solve_state", "step_entry"):
                raise ValueError(
                    "Unknown TKEConfig.tke_n2_evaluation_stage: expected "
                    "'implicit_solve_state' or 'step_entry', got "
                    f"{stage!r}.")
            _n2_mode = getattr(tke_cfg, "n2_mode", "insitu")
            _n2_eos_form = getattr(tke_cfg, "n2_eos_form", "seos")
        elif _gm_carried:
            # stprk3.f90:141-159 builds rn2b at whole-step entry before
            # ldf_slp even when the vertical closure is constant.  The bundle
            # is therefore a stage-program operand, not TKE-owned state.
            _n2_mode = getattr(_gm, "slope_n2", "adiabatic")
            _n2_eos_form = "seos" if _cfg_b.eos == "nemo_seos" else _cfg_b.eos
        else:
            return None
        if _n2_mode != "nemo_bn2":
            raise ValueError(
                "a carried step-entry N2 bundle requires slope/TKE "
                f"n2_mode='nemo_bn2', got {_n2_mode!r}")

        from legoesm.ocean.eos import (
            compute_buoyancy_frequency_nemo_bn2,
            nemo_bn2_depth_ladders,
            nemo_bn2_live_geometry,
            nemo_e3w_from_live_gdept,
            nemo_r3t_stretch,
        )
        from legoesm.ocean.physics.vertical_mixing.tke import TKEEntryN2Bundle
        from legoesm.ocean.vertical import extrapolate_below_seafloor

        _bn2_tracers = self._nemo_ws_test_hooks.bn2_tracer_override
        if _bn2_tracers is None:
            T_now = state.T.data
            S_now = state.S.data
            if getattr(_zc, "is_active", None) is not None:
                T_now = extrapolate_below_seafloor(T_now, _zc)
                S_now = extrapolate_below_seafloor(S_now, _zc)
        else:
            T_now, S_now = (jnp.asarray(value) for value in _bn2_tracers)
            if (T_now.shape != state.T.data.shape
                    or S_now.shape != state.S.data.shape):
                raise ValueError(
                    "the private bn2 tracer override must match state T/S; "
                    f"got T={T_now.shape}, S={S_now.shape}, "
                    f"state={state.T.data.shape}")
        gdept, gdepw, e3w = nemo_bn2_live_geometry(
            _zc, state.eta.data, state.H_bathy.data,
            r3t_evaluation="nemo_reciprocal")
        gdept_0 = getattr(_zc, "nemo_gdept_0", None)
        gdepw_0 = getattr(_zc, "nemo_gdepw_0", None)
        e3t_0 = getattr(_zc, "nemo_e3t_0", None)
        if gdept_0 is None or gdepw_0 is None or e3t_0 is None:
            raise ValueError(
                "tke_n2_evaluation_stage='step_entry' requires raw NEMO "
                "nemo_gdept_0/nemo_gdepw_0/nemo_e3t_0 mesh fields; the "
                "1-D/reconstructed ladder is not a faithful eosbn2/zdftke "
                "operand")
        # The raw NEMO W ladder includes the zero-depth surface point;
        # eosbn2 starts at jk=2, so its first operand is gdepw_0(2).
        gdepw_0 = gdepw_0[..., 1:]
        zrw_stretch = nemo_r3t_stretch(
            _zc, state.eta.data, state.H_bathy.data,
            evaluation="nemo_reciprocal")
        e3w_surface = nemo_e3w_from_live_gdept(
            _zc, gdept, stretch=zrw_stretch, interior=False)[..., :1]
        _n2_kwargs = dict(
            cfg=_cfg_b.eos_nemo_seos,
            g=_cfg_b.constants.g,
            eos_form=_n2_eos_form,
            e3w_int=e3w,
            e3w_source="mesh_reference",
            zrw_evaluation="nemo_literal",
            zrw_gdept_0=gdept_0,
            zrw_gdepw_0=gdepw_0,
            zrw_stretch=zrw_stretch,
        )
        _bn2_selector = self._nemo_ws_test_hooks.bn2_intermediate
        _bn2_override = self._nemo_ws_test_hooks.bn2_alpha_beta_override
        _rn2_result = compute_buoyancy_frequency_nemo_bn2(
            T_now, S_now, gdept, gdepw, **_n2_kwargs,
            _alpha_beta_override=_bn2_override,
            _return_intermediate=_bn2_selector)
        if _bn2_selector:
            rn2, _bn2_intermediate = _rn2_result
        else:
            rn2, _bn2_intermediate = _rn2_result, None

        before = (self._n2_nemo_before_tracers(
            state, z_coord=_zc, config=_cfg_b) if _tke_active else None)
        if before is None:
            rn2b = rn2
        else:
            T_before, S_before = before
            if getattr(_zc, "is_active", None) is not None:
                T_before = extrapolate_below_seafloor(T_before, _zc)
                S_before = extrapolate_below_seafloor(S_before, _zc)
            rn2b = compute_buoyancy_frequency_nemo_bn2(
                T_before, S_before, gdept, gdepw, **_n2_kwargs)
        e3t_0_array = jnp.asarray(e3t_0)
        tmask = getattr(_zc, "is_active", None)
        if tmask is None:
            e3t = e3t_0_array * zrw_stretch[..., jnp.newaxis]
        else:
            # key_qco macro, domzgr_substitute.h90:46/126:
            # E3t_0 * (1 + r3t*tmask).  Dry/pad T slots stay at raw-mesh
            # E3t_0; stretching them changed the ldown carry in shelf columns.
            # where() preserves the already-verified wet multiplication bits.
            e3t = jnp.where(
                jnp.asarray(tmask, dtype=bool),
                e3t_0_array * zrw_stretch[..., jnp.newaxis],
                e3t_0_array)
        return TKEEntryN2Bundle(
            rn2=rn2, rn2b=rn2b, gdepw_Kmm=gdepw, e3w_Kmm=e3w,
            e3t_Kmm=e3t, e3w_surface_Kmm=e3w_surface,
            bn2_intermediate=_bn2_intermediate)

    def _tke_step_entry_p_sh2(
        self, state, *, eta_now=None, u_now=None, v_now=None,
        z_coord=None, config=None, grid=None, return_face_metrics=False,
    ):
        """Freeze NEMO ``p_sh2`` from the selected step-entry face levels."""
        _zc = self.z_coord if z_coord is None else z_coord
        _cfg_b = self.config if config is None else config
        vmix = getattr(getattr(_cfg_b, "physics", None),
                       "vertical_mixing", None)
        if vmix is None or vmix.scheme != "tke":
            return None
        tke_cfg = vmix.tke
        stage = getattr(
            tke_cfg, "tke_shear_evaluation_stage", "implicit_solve_state")
        if stage == "implicit_solve_state":
            return None
        if stage != "step_entry":
            raise ValueError(
                "Unknown TKEConfig.tke_shear_evaluation_stage: expected "
                "'implicit_solve_state' or 'step_entry', got "
                f"{stage!r}.")
        if state.tke_avm is None:
            raise ValueError(
                "tke_shear_evaluation_stage='step_entry' requires carried "
                "state.tke_avm (NEMO avm_k).")

        _u_now = state.u.data if u_now is None else u_now
        _v_now = state.v.data if v_now is None else v_now
        _eta_now = state.eta.data if eta_now is None else eta_now
        from legoesm.ocean.vertical import compute_ocean_jacobian
        J = compute_ocean_jacobian(_eta_now, state.H_bathy.data, _zc)
        dz_half = jnp.broadcast_to(
            _zc.dz_half_ref * J[..., jnp.newaxis],
            state.T.data.shape[:-1] + (_zc.n_levels - 1,))
        shear_disc = getattr(
            tke_cfg, "tke_shear_production", "squared_centered")
        # ``nemo_face_native_nbb2`` is the source-level name adopted by the
        # ORCA2 card for the pre-existing GYRE ``now2`` selector.  Both mean
        # the compiled RK3 call zdf_phy(Nbb,Nbb); keep both card spellings
        # while routing one shared statement below.
        if shear_disc == "nemo_face_native_nbb2":
            shear_disc = "nemo_face_native_now2"
        avm_weighting = getattr(
            tke_cfg, "tke_shear_avm_weighting", "tpoint")
        if shear_disc == "squared_centered":
            if avm_weighting != "tpoint":
                raise ValueError(
                    "step-entry squared_centered shear requires "
                    "tke_shear_avm_weighting='tpoint'.")
            from legoesm.ocean.physics.vertical_mixing._shared import (
                vertical_shear_squared,
            )
            u_cell = 0.5 * (_u_now[:, :-1, :] + _u_now[:, 1:, :])
            v_cell = 0.5 * (_v_now[:-1, :, :] + _v_now[1:, :, :])
            p_sh2 = state.tke_avm.data * vertical_shear_squared(
                u_cell, v_cell, dz_half)
            return (p_sh2, None) if return_face_metrics else p_sh2

        if shear_disc not in ("nemo_face_native", "nemo_face_native_now2"):
            raise ValueError(
                "tke_shear_evaluation_stage='step_entry' supports "
                "'squared_centered', 'nemo_face_native', or "
                f"'nemo_face_native_now2'; got {shear_disc!r}.")
        if avm_weighting != "nemo_face":
            raise ValueError(
                "step-entry face-native shear requires "
                "tke_shear_avm_weighting='nemo_face'.")
        if shear_disc == "nemo_face_native":
            if state.u_before is None or state.v_before is None:
                raise ValueError(
                    "tke_shear_production='nemo_face_native' at step entry "
                    "requires carried state.u_before/v_before face velocities.")
            _u_before = state.u_before.data
            _v_before = state.v_before.data
        else:
            _u_before, _v_before = _u_now, _v_now
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            compute_face_masks_3d,
        )
        from legoesm.ocean.physics.vertical_mixing._shared import (
            avm_weighted_shear_production,
        )
        is_active = getattr(_zc, "is_active", None)
        if is_active is None:
            raise ValueError(
                "step-entry face-native shear requires z_coord.is_active "
                "for NEMO face masks.")
        u_mask, v_mask = compute_face_masks_3d(is_active)
        metric_source = getattr(
            tke_cfg, "tke_shear_metric_source", "tpoint_jacobian")
        if metric_source not in ("tpoint_jacobian", "nemo_qco_live_face"):
            raise ValueError(
                "Unknown TKEConfig.tke_shear_metric_source: expected "
                "'tpoint_jacobian' or 'nemo_qco_live_face', got "
                f"{metric_source!r}.")
        face_metrics = None
        if metric_source == "nemo_qco_live_face":
            if shear_disc == "nemo_face_native_now2":
                # NEMO RK3 calls zdf_phy(kstp, Nbb, Nbb, Nrhs)
                # (stprk3.f90:168): both factors of zdfsh2.f90:83-114 use
                # the step-entry level, so r3u(Kbb) IS r3u(Kmm) and the
                # Kbb face metric is the step-entry one.  Static branch on
                # the variant, not on the presence of a before-state.
                _eta_bef = _eta_now
            else:
                if state.eta_before is None:
                    raise ValueError(
                        "tke_shear_metric_source='nemo_qco_live_face' "
                        "requires state.eta_before for the Kbb face metric.")
                _eta_bef = state.eta_before.data
            e3w0 = getattr(_zc, "nemo_e3w_0", None)
            if e3w0 is None or not bool(getattr(
                    _zc, "nemo_e3w_mesh_reference", False)):
                raise ValueError(
                    "tke_shear_metric_source='nemo_qco_live_face' requires "
                    "the raw mesh nemo_e3w_0 reference field.")
            dtype = state.T.data.dtype
            hu0 = getattr(_zc, "nemo_hu_0", None)
            hv0 = getattr(_zc, "nemo_hv_0", None)
            a_t = getattr(_zc, "nemo_e1e2t", None)
            a_u = getattr(_zc, "nemo_e1e2u", None)
            a_v = getattr(_zc, "nemo_e1e2v", None)
            if any(x is None for x in (hu0, hv0, a_t, a_u, a_v)):
                raise ValueError(
                    "tke_shear_metric_source='nemo_qco_live_face' requires "
                    "raw mesh hu_0/hv_0 and e1e2t/e1e2u/e1e2v fields.")
            hu0 = jnp.asarray(hu0, dtype=dtype)
            hv0 = jnp.asarray(hv0, dtype=dtype)
            a_t = jnp.asarray(a_t, dtype=dtype)
            a_u = jnp.asarray(a_u, dtype=dtype)
            a_v = jnp.asarray(a_v, dtype=dtype)

            def _qco_r3(eta):
                num_u = 0.5 * (
                    a_t * eta + jnp.roll(a_t * eta, -1, axis=1))
                num_v = 0.5 * (
                    a_t * eta + jnp.roll(a_t * eta, -1, axis=0))
                wet_u = hu0 > 0.0
                wet_v = hv0 > 0.0
                wet_u_f = wet_u.astype(dtype)
                wet_v_f = wet_v.astype(dtype)
                # Preserve domqco/domain.F90 operation order exactly:
                # r1_hu_0 = mask/(hu_0 + 1 - mask), then
                # r3u = numerator * r1_hu_0 / e1e2u.  Combining the two
                # divisors changes the last bits and fails the 1e-15 bar.
                r1_hu0 = wet_u_f / (hu0 + 1.0 - wet_u_f)
                r1_hv0 = wet_v_f / (hv0 + 1.0 - wet_v_f)
                r3u_nemo = num_u * r1_hu0 / a_u
                r3v_nemo = num_v * r1_hv0 / a_v
                # NEMO arrays name the east/north face of T(i,j); legoESM
                # arrays name the west/south face.  Prefix the periodic/wall
                # face to convert without changing arithmetic in r3 itself.
                return (
                    jnp.concatenate([r3u_nemo[:, -1:], r3u_nemo], axis=1),
                    jnp.concatenate([r3v_nemo[:1, :], r3v_nemo], axis=0),
                )

            r3un, r3vn = _qco_r3(_eta_now)
            r3ub, r3vb = _qco_r3(_eta_bef)
            # DINO full-step z has e3uw_0 == e3vw_0 == raw mesh e3w_0
            # bit-for-bit; map NEMO east/north-face indexing to legoESM's
            # west/south raw-face arrays before applying each live factor.
            ref = jnp.asarray(e3w0[..., 1:], dtype=dtype)
            ref_u = jnp.concatenate([ref[:, -1:, :], ref], axis=1)
            ref_v = jnp.concatenate([ref[:1, :, :], ref], axis=0)
            face_metrics = (
                ref_u * (1.0 + r3un[..., None]),
                ref_u * (1.0 + r3ub[..., None]),
                ref_v * (1.0 + r3vn[..., None]),
                ref_v * (1.0 + r3vb[..., None]),
            )
        p_sh2 = avm_weighted_shear_production(
            _u_now, _v_now, _u_before, _v_before,
            dz_half, u_mask, v_mask, state.tke_avm.data,
            face_metrics=face_metrics)
        return (p_sh2, face_metrics) if return_face_metrics else p_sh2

    def _tke_realized_kdiss_active(self) -> bool:
        """True iff the post-mixing TKE charges the REALIZED implicit-friction
        dissipation (Veros K_diss_v, friction.py:131-151) instead of the
        pre-solve ``K_M·S²`` (``TKEConfig.shear_production="realized_veros"``;
        static config-only predicate)."""
        if not self._tke_post_mixing_active():
            return False
        tke_cfg = self.config.physics.vertical_mixing.tke
        return (getattr(tke_cfg, "shear_production", "pre_solve")
                == "realized_veros")

    def _tke_advection_active(self) -> bool:
        """True iff the prognostic TKE field is ADVECTED (Veros
        ``enable_tke_superbee_advection``).

        Static Python predicate (config-only): prognostic TKE is on AND
        ``vertical_mixing.tke.advection_scheme != "none"``. Default False ⇒
        the model step traces zero additional ops (bit-identical).
        """
        if not self._tke_prognostic_active():
            return False
        # Superbee TKE advection is a TKE-scheme feature (not CATKE).
        if self.config.physics.vertical_mixing.scheme != "tke":
            return False
        tke_cfg = self.config.physics.vertical_mixing.tke
        return getattr(tke_cfg, "advection_scheme", "none") != "none"

    def _apply_tke_advection(self, state, tke_new, dt, *, grid=None, z_coord=None, config=None):
        """Apply the AB2 advective increment to the freshly-solved TKE (Veros
        ``integrate_tke``'s superbee-advection block, tke.py:286-323).

        ``grid`` (optional, SPMD): default ``None`` → ``self.grid``
        (bit-identical single-device path); a band-local grid is injected
        by a future ``shard_map`` wrapper.

        Computes the W-grid superbee advective tendency ``dtke^n`` of the
        CARRIED ``state.tke`` (Veros tke[tau]) using the PRE-STEP velocities
        ``state.u/v`` (Veros u[tau], the same-step fields the 3-D EKE
        advection uses), then

            tke ← tke_new + dt·((1.5+ε)·dtke^n − (0.5+ε)·dtke^{n-1})

        with ``dt`` the TRACER timestep (Veros multiplies by
        ``settings.dt_tracer``, tke.py:318 — NOT dt_tke=dt_mom, which only
        drives the implicit solve, tke.py:137) and ``ε = config.ab2_epsilon``
        (the model's one AB2 epsilon; Veros AB_eps). ``dtke^{n-1}`` is the
        ``state.dtke`` carry — zero when None (first step), exactly Veros's
        zero-initialised ``dtke[taum1]``. No positivity floor is applied
        (Veros applies none after this add; every TKE consumer floors
        internally).

        Returns ``(tke_out, dtke_field)`` where ``dtke_field`` is the new
        carry. Raises ``ValueError`` on an unknown scheme literal (dispatch
        hardening; construction validation is the first gate).
        """
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        from legoesm.ocean.advection import (
            wgrid_advection_tendency_latlon_cgrid,
        )

        _grid = grid if grid is not None else self.grid
        tke_cfg = _cfg_b.physics.vertical_mixing.tke
        scheme = getattr(tke_cfg, "advection_scheme", "none")
        if scheme != "superbee":
            raise ValueError(
                f"Unknown TKE advection_scheme {scheme!r}; expected "
                f"'superbee' (or 'none', which never reaches this dispatch).")
        dtype = tke_new.dtype
        lm = state.land_mask.data
        if state.tke is not None:
            tke_tau = state.tke.data
        else:
            # Cold start (direct step without integrate_scan pre-seeding):
            # advect the same background-seeded field the implicit solve
            # seeds (a constant field ⇒ ~zero tendency up to the W-grid
            # continuity residual absorbed at interface 0).
            nlev = state.T.data.shape[-1]
            tke_tau = jnp.where(
                lm[:, :, jnp.newaxis] > 0.5, tke_cfg.tke_background, 0.0,
            ).astype(dtype) * jnp.ones((1, 1, nlev - 1), dtype=dtype)
        # Variable-bathymetry guard (partial-cell coords): the advecting u/v
        # must be zero at sub-seafloor levels (the barotropic correction
        # writes U_bar at EVERY level of a face column, so the raw arrays
        # carry O(0.1–1 m/s) below the seafloor of shallow columns) and the
        # tendency must vanish at dry interfaces — Veros's maskU/maskV/maskW
        # in calculate_velocity_on_wgrid + the dtke assembly.  Without this,
        # shallow shelf columns accumulate ±1e7 m²/s² TKE within ~20 steps
        # (global_4deg 78°N blowup).  Flat-bottom (pure z-star): no-op.
        u_adv, v_adv = state.u.data, state.v.data
        _wet_if_adv = None
        if isinstance(_zc, OceanPartialCellCoordinate):
            _um3, _vm3 = compute_face_masks_3d(
                _zc.is_active, _grid)
            u_adv = u_adv * _um3.astype(u_adv.dtype)
            v_adv = v_adv * _vm3.astype(v_adv.dtype)
            _wet_if_adv = _zc.is_active.astype(dtype)[..., 1:]
            tke_tau = tke_tau * _wet_if_adv
        dtke_now = wgrid_advection_tendency_latlon_cgrid(
            tke_tau, u_adv, v_adv, _grid,
            jnp.asarray(_zc.dz_ref), dt, lm,
            state.u_mask.data, state.v_mask.data,
        )
        dtke_now = jax.lax.convert_element_type(dtke_now, dtype)
        if _wet_if_adv is not None:
            dtke_now = dtke_now * _wet_if_adv
        dtke_prev = (state.dtke.data.astype(dtype)
                     if state.dtke is not None else jnp.zeros_like(dtke_now))
        eps = _cfg_b.ab2_epsilon
        # (#517 item 8: shared ab2_blend; eps verbatim → bit-identical.)
        tke_out = tke_new + dt * ab2_blend(dtke_now, dtke_prev, eps)
        dtke_field = Field(data=dtke_now, name="dtke",
                           dims=("lat", "lon", "level"), units="m^2/s^3")
        return tke_out, dtke_field

    def _assemble_tke_source(self, state, state_new, tend, z_coord=None, config=None):
        """Assemble the prognostic-TKE energy-recycling source ``forc`` [m²/s³].

        Mirrors Veros integrate_tke ``forc = ... + eke_diss_iw + K_diss_bot``
        (the ACC short-cut without idemix). Both terms are ≥ 0 (recycled,
        already-dissipated mechanical energy) and live at the interior
        interfaces ``(n_lat, n_lon, nlev-1)`` — the same W-grid the prognostic
        TKE field lives on. Each is gated by its own config flag
        (``source_eke_diss`` / ``source_bottom_drag_diss``); when both are off
        the source is ``None`` ⇒ bit-identical to the closure without recycled
        sources.

        - ``eke_diss``: the EKE dissipation rate (Veros ``eke_diss_iw``). The
          3-D EKE step (``_eke_3d_step``) runs in the GM/Redi stage of THIS
          model step — BEFORE this implicit-mixing TKE solve — so ``state_new``
          already carries this step's ``eke_diss`` (matching Veros's same-step
          eke→tke ordering, veros.py:277,285: NO lag). Falls back to the carried
          ``state.eke_diss`` if ``state_new`` has none yet (e.g. the 2-D EKE
          path, which does not produce the 3-D dissipation field — then the
          carried field is a documented ONE-STEP LAG, or ``None`` ⇒ no source).
        - ``K_diss_bot``: this step's bottom-drag KE extraction, surfaced as
          the tendency diagnostic ``tend.K_diss_bot``.
        """
        # z_coord accepted for uniform SPMD forwarding; this helper reads no vertical geometry directly.
        _cfg_b = self.config if config is None else config  # SPMD band override
        tke_cfg = _cfg_b.physics.vertical_mixing.tke
        source = None

        def _accum(src, field):
            return field if src is None else (src + field)

        if getattr(tke_cfg, "source_eke_diss", False):
            _eke_diss = state_new.eke_diss
            if _eke_diss is None:
                _eke_diss = state.eke_diss
            if _eke_diss is not None:
                source = _accum(source, jnp.maximum(_eke_diss.data, 0.0))
        if getattr(tke_cfg, "source_bottom_drag_diss", False):
            if tend.K_diss_bot is not None:
                source = _accum(source, jnp.maximum(tend.K_diss_bot.data, 0.0))
        return source

    def _eke_3d_step(
        self,
        state: LatLonCGridOceanState,
        state_new: LatLonCGridOceanState,
        T_mid: jnp.ndarray,
        S_mid: jnp.ndarray,
        gm_cfg,
        eke_cfg,
        lm: jnp.ndarray,
        A_v_phys,
        dt: float,
        *,
        Ah_visc_u=None,
        Ah_visc_v=None,
        Ah_kediss_cell=None,
        grid=None,
        resfn_scale=None,
        z_coord=None, config=None,
    ) -> tuple:
        """One step of the 3-D (depth-resolved) prognostic-EKE closure.

        ``grid`` (optional, SPMD): default ``None`` → ``self.grid``
        (bit-identical single-device path); a band-local grid is injected
        by a future ``shard_map`` wrapper.

        ``resfn_scale`` (optional, codex MED-3 r2): the 2-D Hallberg
        resolution factor ``f_res`` the step computed when
        ``gm_cfg.resolution_function`` is on.  The GM/Redi tracer tendency
        applies ``kappa_eff = f_res·kappa`` itself, so the RETURNED
        ``kappa_gm_override`` stays RAW here; ``f_res`` scales only the
        GM-DERIVED EKE production (parameterized ``kappa·sigma²`` via
        ``production_scale``; realized skew conversions via the scaled kappa
        handed to the conversion builders) so the eddy-energy budget matches
        the applied coefficient.  The Redi-side ``kappa_redi_w`` (and hence
        ``-P_diss_iso``) stays raw — the taper is GM-only.  ``None``
        (default off) ⇒ bit-identical.

        The eddy-energy field ``E`` lives on the ``nlev-1`` interior interfaces
        (the W-grid), matching Veros's 3-D ``vs.eke``.  Returns
        ``(eke_field_new, kappa_gm_override)`` where ``eke_field_new`` is the
        updated 3-D ``Field (n_lat, n_lon, nlev-1)`` (floored at ``e_min``,
        masked to wet columns) and ``kappa_gm_override`` is the 3-D
        interface GM coefficient ``kappa_GM(z) = c_k·L(z)·√E`` fed to the GM/Redi
        tracer tendency (Stage 3 consumes the 3-D interface kappa).

        Operator-split (each sub-step unconditionally stable / positivity-
        respecting), applied in the order:

          1. per-interface horizontal transport (explicit upwind advection +
             lateral diffusion), advecting ``E`` by the flow AT EACH INTERFACE
             (the full-level u/v vertically averaged to the W-grid),
          2. implicit (backward-Euler) vertical EKE diffusion ``K = alpha_eke·A_v``,
          3. semi-implicit local source/sink ``P(z) - eps(z)`` (positivity-
             preserving, no clip).

        The source/sink is applied LAST so the depth-resolved production
        ``P(z) = kappa_GM(z)·sigma(z)²`` enters after transport+diffusion have
        redistributed ``E`` — mirroring the 2-D path's (transport → source)
        order, with the vertical diffusion inserted between (the 2-D path has no
        vertical operator).  ``E`` is finally floored at ``e_min`` and masked.

        EKE-source augmentation (Veros apples-to-apples; both default off via
        ``eke_cfg`` flags ⇒ the existing source is bit-identical):

          - ``source_kdiss_h`` — add the mean-KE removed by the harmonic LATERAL
            viscosity (Veros ``K_diss_h``) as a non-negative source, built from the
            harmonic-viscosity momentum tendency ``A_h∇²(u,v)`` (``Ah_visc_u/v``,
            computed once by the tendency function) via
            :func:`...gm_redi_latlon_cgrid.harmonic_lateral_kediss_eke_source`.
          - ``gm_source_mode="realized"`` — REPLACE the parameterized GM conversion
            ``kappa_GM·sigma²`` with the realized GM-skew buoyancy conversion
            ``-(g/ρ₀)∇ρ·F_skew`` (Veros ``-P_diss_skew``) via
            :func:`...gm_redi_latlon_cgrid.compute_realized_gm_skew_conversion`.

        Both enter the EXPLICIT production of the semi-implicit update, so
        positivity-by-construction is preserved (both are ≥ 0).

        W-grid metrics (subtle — documented):

        - ``dz_w`` = the W-cell thicknesses = ``build_dz_half(dz_cell)`` =
          ``0.5(dz_k + dz_{k+1})`` (the distance between adjacent T-cell centres
          = Veros's ``dzw``; the divergence divisor in the implicit solve, and
          the weight in the conserved column integral ``Σ E·dz_w``).
        - ``dz_half_w`` = ``build_dz_half(dz_w)`` = the midpoint spacing between
          adjacent W-cell centres (the flux divisor in the implicit solve).  The
          legoESM W-grid is treated as an independent ``M=nlev-1``-level column
          with zero-flux BCs at its top/bottom, so this midpoint spacing is the
          self-consistent flux metric (it equals Veros's ``dzt`` flux divisor on
          a uniform grid; the 2nd-order metric difference on a stretched grid is
          within the documented operator-split treatment).
        - ``A_v`` at the ``M-1`` interior W-interfaces: the interior W-interface
          ``k`` sits at the centre of T-cell ``k+1``, so the vertical viscosity
          there is ``A_v_cell[..., 1:nlev-1]`` (Veros's
          ``0.5(kappaM[k]+kappaM[k+1])`` averaged to the W-interfaces; the
          jacobian-scaled cell-centred ``A_v`` is the same profile the implicit
          momentum solve uses, plus the config background floor).
        """
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        from legoesm.ocean.physics.vertical_mixing import build_dz_half
        from legoesm.ocean.vertical import compute_ocean_jacobian

        _grid = grid if grid is not None else self.grid
        dtype = T_mid.dtype
        nlev = T_mid.shape[-1]
        # E on the interior interfaces (W-grid).  Seeded to e_min if absent so
        # the None -> Field transition (which would break the scan-carry pytree)
        # never happens inside the step when integrate_scan pre-seeded it.
        if state.eke is not None:
            E = state.eke.data
        else:
            E = jnp.full(lm.shape + (nlev - 1,), eke_cfg.e_min, dtype=dtype)

        # Depth-resolved kappa_GM(z), Eady growth sigma(z), mixing length L(z) at
        # the interior interfaces (Stage 1 depth_resolved path).
        kappa_gm_override, sigma3, L3 = compute_eke_step_kappa(
            T_mid, S_mid, state_new.eta.data, state_new.H_bathy.data,
            E, _grid, _zc, gm_cfg,
            eos=_cfg_b.eos, eos_linear=_cfg_b.eos_linear,
            mask=lm,
            rho_0=_cfg_b.constants.rho_0,
            g=_cfg_b.constants.g,
            omega=_cfg_b.constants.Omega,
            r_earth=_cfg_b.constants.R_earth,
            depth_resolved=True,
        )

        # Per-interface advecting flow: full-level u/v (nlev) vertically averaged
        # to the nlev-1 interior interfaces (0.5*(u[...,k]+u[...,k+1])).  Face
        # masks (2-D) are applied so transport sees no flow through walls; the
        # 3-D transport broadcasts them over the level axis internally.
        u_mask = state.u_mask.data
        v_mask = state.v_mask.data
        # Variable-bathymetry guard (partial-cell coords): zero the advecting
        # u/v at sub-seafloor levels (the barotropic correction writes U_bar at
        # every level of a face column — see _apply_tke_advection) and keep the
        # eddy energy off the dry interfaces.  Veros maskU/maskV/maskW.  Pure
        # z-star: no-op (masks all ones).
        _u_lvl, _v_lvl = state.u.data, state.v.data
        _wet_if_eke = None
        if isinstance(_zc, OceanPartialCellCoordinate):
            _um3, _vm3 = compute_face_masks_3d(
                _zc.is_active, _grid)
            _u_lvl = _u_lvl * _um3.astype(_u_lvl.dtype)
            _v_lvl = _v_lvl * _vm3.astype(_v_lvl.dtype)
            _wet_if_eke = _zc.is_active.astype(dtype)[..., 1:]
            E = E * _wet_if_eke
        U_z = 0.5 * (_u_lvl[..., :-1] + _u_lvl[..., 1:])
        V_z = 0.5 * (_v_lvl[..., :-1] + _v_lvl[..., 1:])
        U_z = U_z * u_mask[:, :, jnp.newaxis]
        V_z = V_z * v_mask[:, :, jnp.newaxis]

        # (1) Per-interface horizontal transport (explicit).
        E = E + dt * eke_3d_horizontal_transport(
            E, U_z, V_z, _grid, eke_cfg, lm, u_mask, v_mask,
        )
        if _wet_if_eke is not None:
            E = E * _wet_if_eke

        # (2) Implicit vertical EKE diffusion on the W-grid column.
        # W-cell thicknesses dz_w (= Veros dzw) and the interior-W-interface
        # viscosity A_v, from the jacobian-scaled cell metrics + the physics A_v.
        J_cell = compute_ocean_jacobian(
            state.eta.data, state.H_bathy.data, _zc,
        )
        # dz metrics in the field dtype so the implicit solve stays consistent
        # (dz_ref is f64 while the eddy-energy field runs at the storage policy's
        # dtype — cast to E's dtype to avoid a f64->f32 scatter cast).
        dz_cell = (_zc.dz_ref * J_cell[..., jnp.newaxis]).astype(dtype)
        dz_w = build_dz_half(dz_cell)                              # (..., nlev-1) = M
        dz_half_w = build_dz_half(dz_w)                            # (..., nlev-2) = M-1
        # A_v at the M-1 interior W-interfaces for the EKE vertical diffusion --
        # handles A_v_phys cell-centred (nlev) / at T-interfaces (nlev-1) / None.
        # KNOWN LIMITATION (codex MED-2 r2): whenever the physics does NOT
        # surface A_v on the tendencies (post-mixing TKE, lat-dependent
        # constant background), this EKE-energy smoothing falls back to the
        # UNIFORM config.A_v — not the scheme's K profile, which is only
        # assembled later inside _apply_implicit_vertical_mixing.  A
        # second-order eddy-ENERGY diffusion coefficient, not the momentum/
        # tracer mixing itself; threading the authoritative fallback profile
        # here needs a step-order change (follow-up if EKE is ever combined
        # with a fallback-only vmix scheme in production).
        A_v_w = _eke_av_at_interior_wfaces(
            A_v_phys, jnp.asarray(_cfg_b.A_v, dtype=dtype), nlev,
            dz_cell.shape[:-1],
        )
        E = eke_3d_vertical_diffusion(E, A_v_w, dz_w, dz_half_w, dt, eke_cfg)

        # (3) Semi-implicit local source/sink (positivity-preserving, no clip).
        # EKE-source augmentation (Veros apples-to-apples; both default off ⇒ the
        # production is bit-identical to the parameterized kappa_GM·sigma² closure):
        #   - gm_source_mode="realized": REPLACE kappa_GM·sigma² with the realized
        #     GM-skew buoyancy conversion -(g/ρ₀)∇ρ·F_skew (Veros -P_diss_skew),
        #     built from the SAME per-triad W-face slopes the GM tracer flux uses
        #     and the SAME 3-D kappa_GM(z) override (kappa_gm_override).
        #   - source_kdiss_h: ADD the mean-KE removed by the harmonic lateral
        #     viscosity (Veros K_diss_h), from the A_h∇²(u,v) tendency.
        # Both are ≥ 0 and enter the EXPLICIT production, so the semi-implicit
        # update stays positivity-preserving (numerator ≥ 0, denominator ≥ 1).
        production_override = None
        signed_source = None
        clamp_production = True
        # Resolution-function coupling: broadcast the 2-D f_res over the
        # W-grid level axis; the realized conversions are LINEAR in the kappa
        # they are handed, so passing the scaled kappa yields exactly the
        # conversion of the f_res-scaled flux the tracer path applies. The
        # returned kappa_gm_override stays RAW (see docstring).
        _scale_w = (None if resfn_scale is None
                    else resfn_scale[..., jnp.newaxis])
        _kappa_gm_src = (kappa_gm_override if _scale_w is None
                         else kappa_gm_override * _scale_w)
        if eke_cfg.gm_source_mode == "realized":
            production_override = compute_realized_gm_skew_conversion(
                T_mid, S_mid, state_new.eta.data, state_new.H_bathy.data,
                _grid, _zc, gm_cfg, _kappa_gm_src,
                eos=_cfg_b.eos, eos_linear=_cfg_b.eos_linear,
                mask=lm,
                rho_0=_cfg_b.constants.rho_0, g=_cfg_b.constants.g,
            )
        elif eke_cfg.gm_source_mode == "realized_signed":
            # Literal SIGNED Veros conversions: -P_diss_skew (source) and, when
            # source_p_diss_iso, -P_diss_iso (Redi sink). The signed skew replaces
            # the parameterized production (NOT clamped — folded semi-implicitly);
            # the iso sink is added to the signed source (it is ≤ 0 mostly, so its
            # negative part folds into the implicit factor, keeping E ≥ e_min).
            # K_iso = K_gm coupling (enable_eke_isopycnal_diffusion): the Redi
            # diffusivity follows the prognostic kappa_GM(z).
            # kappa_redi_w stays the RAW kappa: the K_iso=K_gm Redi override
            # the tracer path consumes is UNSCALED (GM-only taper), so its
            # realized -P_diss_iso must be built from the same raw kappa.
            kappa_redi_w = (kappa_gm_override
                            if eke_cfg.isopycnal_diffusion else None)
            _signed_kwargs = dict(
                eos=_cfg_b.eos, eos_linear=_cfg_b.eos_linear,
                mask=lm, u_mask=state.u_mask.data, v_mask=state.v_mask.data,
                rho_0=_cfg_b.constants.rho_0, g=_cfg_b.constants.g,
                dt=dt,
            )
            if (_scale_w is not None and eke_cfg.source_p_diss_iso
                    and kappa_redi_w is None):
                # Codex MED-3 r2 finding 1: with the UNCOUPLED iso sink
                # (isopycnal_diffusion=False), the callee's kappa_redi_w=None
                # fallback ALIASES its kappa_gm_w argument — which is now the
                # f_res-SCALED kappa — silently scaling the Redi-side
                # -P_diss_iso the GM-only taper must not touch. Split the
                # call: the skew conversion from the SCALED kappa, the iso
                # conversion from the RAW kappa (the exact VALUE the legacy
                # fallback used). Passing kappa_redi_w=raw in ONE call would
                # instead flip the _skew_ddk double_redi_diagonal fallback
                # (gm_redi_latlon_cgrid) off cfg.kappa_Redi — wrong there.
                # Cost: the shared EOS/enthalpy preamble runs twice, only in
                # this opt-in corner. Byte-identical when the resolution
                # function is off (single legacy call below).
                neg_skew, _ = compute_realized_signed_conversions(
                    T_mid, S_mid, state_new.eta.data, state_new.H_bathy.data,
                    _grid, _zc, gm_cfg, _kappa_gm_src,
                    want_skew=True, want_iso=False, kappa_redi_w=None,
                    **_signed_kwargs,
                )
                _, neg_iso = compute_realized_signed_conversions(
                    T_mid, S_mid, state_new.eta.data, state_new.H_bathy.data,
                    _grid, _zc, gm_cfg, kappa_gm_override,
                    want_skew=False, want_iso=True, kappa_redi_w=None,
                    **_signed_kwargs,
                )
            else:
                neg_skew, neg_iso = compute_realized_signed_conversions(
                    T_mid, S_mid, state_new.eta.data, state_new.H_bathy.data,
                    _grid, _zc, gm_cfg, _kappa_gm_src,
                    want_skew=True, want_iso=eke_cfg.source_p_diss_iso,
                    kappa_redi_w=kappa_redi_w,
                    **_signed_kwargs,
                )
            production_override = neg_skew
            clamp_production = False
            if eke_cfg.source_p_diss_iso:
                # -P_diss_iso is the Redi APE-dissipation; SUBTRACT it from the EKE
                # source (Veros forc has -P_diss_iso, mostly < 0 ⇒ a sink). neg_iso
                # IS -P_diss_iso, so it is added directly as a signed source.
                signed_source = neg_iso
        extra_source = None
        if eke_cfg.source_kdiss_h and Ah_visc_u is not None:
            # Flux form (kdiss_h_flux_form=True): pass the pre-computed positive-
            # definite cell-centre dissipation density (Ah_kediss_cell) so the
            # source builder maps it to the W-grid with NO clamp.  Dynamical form
            # (Ah_kediss_cell is None): the clamped -u·A_h∇²u path.
            _kdiss_cell = (Ah_kediss_cell.data if Ah_kediss_cell is not None
                           else None)
            extra_source = harmonic_lateral_kediss_eke_source(
                Ah_visc_u.data, Ah_visc_v.data, state.u.data, state.v.data,
                _grid, lm, kdiss_h_cell=_kdiss_cell,
            )
        E, eke_diss_w = eke_apply_local_source(
            E, sigma3, L3, eke_cfg, dt,
            production_override=production_override, extra_source=extra_source,
            signed_source=signed_source, clamp_production=clamp_production,
            return_dissipation=True,
            production_scale=_scale_w,
        )

        # Floor at e_min on wet columns, zero on land (kappa_gm_override is
        # already wet-masked by compute_eke_kappa_gm).  Cast back to the field
        # dtype: the source/sink promotes through the f64 sigma(z)/L(z), but the
        # stored eke must keep the storage-policy dtype so the scan-carry pytree
        # is dtype-stable (the step's final cast_pytree would also enforce this,
        # but keep this code path self-consistently typed).
        lm3 = lm[:, :, jnp.newaxis]
        # The source/sink promotes E through the f64 sigma(z)/L(z); cast the final
        # field back to the storage dtype EXPLICITLY (jax.lax.convert_element_type,
        # not .astype) so the scan-carry pytree stays dtype-stable WITHOUT JAX's
        # implicit-downcast FutureWarning (which a future JAX makes an error).
        E_new = jax.lax.convert_element_type(
            jnp.where(lm3 > 0.5, jnp.maximum(E, eke_cfg.e_min), 0.0), dtype,
        )
        if _wet_if_eke is not None:
            # Keep the eddy energy off dry interfaces (Veros maskW) on
            # partial-cell coords; no-op on flat-bottom.
            E_new = E_new * jax.lax.convert_element_type(_wet_if_eke, dtype)
        eke_new = Field(data=E_new, name="eke",
                        dims=("lat", "lon", "level"), units="m^2/s^2")
        # EKE dissipation rate (Veros eke_diss_iw) at the interior interfaces,
        # cast to the storage dtype + wet-masked, for the prognostic-TKE source
        # (consumed ONE STEP LATER — the TKE K-profile solve precedes this EKE
        # step in the legoESM model step). Field so the scan-carry pytree is
        # stable; ≥ 0 by construction.
        eke_diss_new = Field(
            data=jax.lax.convert_element_type(
                jnp.where(lm3 > 0.5, jnp.maximum(eke_diss_w, 0.0), 0.0), dtype),
            name="eke_diss", dims=("lat", "lon", "level"), units="m^2/s^3")
        return eke_new, kappa_gm_override, eke_diss_new

    def _apply_implicit_vertical_mixing(
        self,
        state: LatLonCGridOceanState,
        dt: float,
        surface_forcing,
        K_v_phys=None,
        A_v_phys=None,
        K33_iso=None,
        *,
        dt_mom=None,
        tke_rn_dt=None,
        surface_tracer_forcing=None,
        tracer_source=None,
        do_tracers: bool = True,
        do_momentum: bool = True,
        tke_old=None,
        tke_source=None,
        return_tke: bool = False,
        return_tke_entry: bool = False,
        K_diss_v_w=None,
        return_K_diss_v: bool = False,
        return_K_profiles: bool = False,
        return_tracer_solve_trace: bool = False,
        effective_K_test_override=None,
        grid=None,
        n2_tracers=None,
        n2_tracers_before=None,
        tke_n2_bundle=None,
        eta_now=None,
        u_now=None,
        v_now=None,
        nemo_tracer_content_rhs=None,
        nemo_aimp_tracer_w=None,
        nemo_aimp_momentum_w_u=None,
        nemo_aimp_momentum_w_v=None,
        z_coord=None, config=None, iwm_fields=None,
    ) -> LatLonCGridOceanState:
        """Backward-Euler vertical diffusion for ``u, v, T, S``.

        Uses the K_v / A_v profiles already computed by the physics
        function (passed via ``K_v_phys`` / ``A_v_phys``), adds the
        ``LatLonCGridOceanConfig.A_v`` / ``K_v`` background floors,
        and applies one tridiagonal solve per column to each prognostic
        field.  The solver enforces zero-flux boundary conditions, so
        the column-mean (and hence the barotropic mode for u, v) is
        preserved exactly.

        When K_v_phys / A_v_phys are None (no physics function, or
        physics that doesn't produce K profiles), falls back to
        ``compute_vertical_K_profiles`` for a fresh computation.

        ``dt`` is the TRACER timestep (implicit vertical DIFFUSION of T, S);
        ``dt_mom`` (default ``dt``) is the MOMENTUM timestep (implicit vertical
        FRICTION of u, v).  They differ only under asynchronous
        ``dt_mom_ratio != 1.0`` stepping — Veros applies implicit friction on
        dt_mom (friction.py) and implicit diffusion on dt_tracer
        (thermodynamics.py).  ``dt_mom is None`` ⇒ ``dt`` ⇒ bit-identical.

        ``surface_tracer_forcing`` (a :class:`SurfaceTracerForcing` or ``None``)
        is the WITHHELD surface TRACER forcing rate (restoring + q_net +
        shortwave), populated only under ``config.surface_forcing_implicit``.
        When present its ``dt·rate`` (masked) is added to the T/S solve INPUT
        BEFORE the tridiagonal solve, realising the backward-Euler RHS-source
        identity ``(I − dt·L)·X_new = X_old + dt·S_surf`` at weight 1.0 — Veros's
        implicit surface-forcing placement (``core/thermodynamics.py``).  ``dt``
        here is dt_tracer (the tracer timestep), matching Veros.  ``None`` ⇒
        no surface source ⇒ bit-identical.

        ``tracer_source`` (same container type, or ``None``) is the WITHHELD
        full-COLUMN tracer source rate — the implicit SPONGE placement
        (``config.sponge_forcing_implicit``, EXT-N2; Veros ``tempsalt_sources``,
        ``core/thermodynamics.py:419`` → ``core/diffusion.py:132-141``).  It is
        added to the solve INPUT exactly like ``surface_tracer_forcing``
        (``dt·rate`` at weight 1.0, dt = dt_tracer) but — the documented
        EXT-N2 caveat — it is EXCLUDED from the post-mixing TKE surface
        buoyancy-flux reconstruction, which column-sums only
        ``surface_tracer_forcing`` (Veros keeps tempsalt_sources out of
        ``forc_rho_surface``).  ``None`` ⇒ bit-identical.

        ``do_tracers`` / ``do_momentum`` (static Python bools) select which
        prognostic fields the solve acts on — the additive momentum-friction
        placement (``config.momentum_friction_additive``) evaluates the
        MOMENTUM solve on the pre-step state u^n and the TRACER solve on the
        AB2 state, in two separate calls.  Defaults (both True) ⇒ the original
        combined solve ⇒ bit-identical.

        ``tke_old`` / ``tke_source`` / ``return_tke`` drive the PROGNOSTIC TKE
        carry (``vertical_mixing.tke.prognostic=True``). ``tke_old`` is the TKE
        field carried on ``state.tke`` (interior interfaces); ``tke_source`` is
        the additive energy-recycling source ``forc`` (eke_diss_iw + K_diss_bot)
        at the interior interfaces. The prognostic TKE solve uses ``dt_mom`` as
        its step (Veros tke.py:137 ``dt_tke = dt_mom``). When ``return_tke`` is
        True this method returns ``(state_new, tke_new)`` where ``tke_new`` is
        the updated TKE field (``None`` unless the prognostic TKE scheme is
        active). Defaults (``return_tke=False``) ⇒ returns the state only ⇒
        bit-identical.

        POST-MIXING TKE (``TKEConfig.buoyancy_timing="post_mixing_veros"``,
        the Veros step order): the fallback K-profile computation returns the
        phase-1 :class:`TKEPostMixingContext` (kappa from the CARRIED TKE =
        Veros set_tke_diffusivities at tau), the tracer/momentum solves run
        on those profiles, and the TKE budget is then solved HERE, AFTER the
        tracer solve — charging the POST-mixing N² (Veros Nsqr[taup1],
        thermodynamics.py:385) plus the surface buoyancy-flux ``P_diss_v``
        slot (thermodynamics.py:386-388).  ``K_diss_v_w`` is an optional
        PRE-COMPUTED realized friction-dissipation field (interior
        interfaces, cell-centred) for ``shear_production="realized_veros"``
        when this call does not solve momentum itself (the additive-friction
        AB2 split); ``return_K_diss_v=True`` (momentum-only call,
        ``return_tke=False``) makes the method return
        ``(state_new, K_diss_v)`` so ``_ab2_step`` can thread it into the
        tracer call.  All defaults ⇒ bit-identical.

        Called only when ``config.implicit_vertical_mixing == True``.

        ``grid`` (optional, SPMD): default ``None`` → ``self.grid``
        (bit-identical single-device path); a band-local grid is injected
        by a future ``shard_map`` wrapper.
        """
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        _iwm = (self._iwm_forcing if iwm_fields is None else iwm_fields)  # SPMD band override
        _zdf_momentum_observer = self._nemo_ws_test_hooks.zdf_momentum_observer
        # Argument validation at ENTRY, not inside the drag branch below: one
        # component of the now-level velocity without the other would build the
        # rate's |U| from two time levels, and a caller that gets it wrong with
        # do_momentum=False or the drag flag off deserves the error just as
        # much (review N4).  Static Python args -- nothing is traced.
        if (u_now is None) != (v_now is None):
            raise ValueError(
                "implicit vertical mixing's now-level drag velocity must "
                "supply BOTH u_now and v_now or NEITHER (one alone mixes "
                "time levels inside one |U|); got "
                f"u_now={'set' if u_now is not None else None}, "
                f"v_now={'set' if v_now is not None else None}.")

        if dt_mom is None:
            dt_mom = dt
        _grid = grid if grid is not None else self.grid
        if return_K_diss_v and return_tke:
            raise ValueError(
                "_apply_implicit_vertical_mixing: return_K_diss_v is for the "
                "momentum-only call (return_tke must be False).")
        tke_new = None
        _tke_coeff_new = None
        _tke_entry_used = None
        _tke_statement_trace_used = None
        _tke_shear_face_metrics_used = None
        _tracer_solve_trace = None
        _trace_heat_K = None
        _trace_isoneutral_K = None
        _post_mixing = self._tke_post_mixing_active()
        _tke_ctx = None
        from legoesm.ocean.physics.vertical_mixing import (
            implicit_vertical_diffusion_ocean_batched, build_dz_half,
            compute_vertical_K_profiles, nemo_e3w_kmm,
        )
        from legoesm.ocean.eos import nemo_r3t_stretch
        from legoesm.ocean.vertical import compute_ocean_jacobian
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            interp_cell_to_uface,
        )

        # MED-2 (codex batch2 BLOCKER): the latitude-dependent constant
        # background must come from the fallback recompute below.  A surfaced
        # K/A (e.g. from enhanced_diffusion convection) would otherwise take
        # the fast path, which ADDS the model floors and DROPS the Gregg
        # field entirely.  The fallback SUPERSEDES (does not double-count)
        # any surfaced convection profile: it re-diagnoses convection on the
        # CURRENT (post-advection) state with the model EOS — the same
        # convention every fallback-path scheme uses; the pre-step surfaced
        # diagnosis is simply discarded.  Static config gate (pure Python
        # bool) — no traced branch.
        if self._lat_dependent_constant_vmix():
            K_v_phys = None
            A_v_phys = None

        if K_v_phys is not None and A_v_phys is not None:
            # Fast path: use K profiles already computed by the physics
            # function, just add the config background floors.
            if _post_mixing and do_tracers:
                raise ValueError(
                    "buoyancy_timing='post_mixing_veros' requires the "
                    "fallback K-profile path (the TKE scheme must not "
                    "surface K_v/A_v on the tendencies): the post-mixing "
                    "TKE solve needs the phase-1 context from "
                    "compute_vertical_K_profiles.")
            state.T.data.shape[-1]
            dtype = state.T.data.dtype
            # NEMO zdfevd REPLACES the assembled coefficient where its trigger
            # fires (zdfevd.f90:107-110, run at zdfphy.f90:359 AFTER the
            # background copy at :348-351), so on a card that states
            # ``evd_composition="nemo_replace"`` the fired interfaces carry
            # rn_evd ALONE -- not rn_evd plus the namelist background.  Every
            # other card keeps the historical sum.  The fallback path does the
            # same composition in k_profiles.compute_vertical_K_profiles.
            _conv_fast = getattr(
                getattr(_cfg_b, "physics", None), "convection", None)
            _evd_fast = None
            if getattr(_conv_fast, "scheme", "none") == "enhanced_diffusion":
                from legoesm.ocean.physics.convection.enhanced_diffusion import (
                    compose_evd_coefficient, resolve_evd_composition,
                )
                _ed_fast = _conv_fast.enhanced_diffusion
                if resolve_evd_composition(_ed_fast) == "nemo_replace":
                    _evd_fast = _ed_fast
            if _evd_fast is None:
                K_v_cell = K_v_phys + jnp.asarray(_cfg_b.K_v, dtype=dtype)
                A_v_cell = A_v_phys + jnp.asarray(_cfg_b.A_v, dtype=dtype)
            else:
                # Fail closed: on THIS path the surfaced K/A are the
                # enhanced-diffusion scheme's own, which is only true when no
                # closure also surfaces one.  tke/catke take the fallback
                # path (combined.make_ocean_physics withholds EVD for them);
                # anything else would have the replace overwrite a closure's
                # coefficient it never saw.
                _vmix_fast = getattr(
                    getattr(_cfg_b, "physics", None), "vertical_mixing", None)
                if getattr(_vmix_fast, "scheme", "none") != "none":
                    raise ValueError(
                        'EnhancedDiffusionConfig.evd_composition='
                        '"nemo_replace" on the physics-provided-K path is '
                        "only defined when the card selects no vertical-"
                        "mixing closure (vertical_mixing.scheme='none'); got "
                        f"{getattr(_vmix_fast, 'scheme', None)!r}. tke/catke "
                        "compose EVD inside compute_vertical_K_profiles.")
                _K_bg_field = jnp.full_like(
                    K_v_phys, jnp.asarray(_cfg_b.K_v, dtype=dtype))
                _A_bg_field = jnp.full_like(
                    A_v_phys, jnp.asarray(_cfg_b.A_v, dtype=dtype))
                K_v_cell = compose_evd_coefficient(
                    _K_bg_field, K_v_phys, "nemo_replace",
                    convective=_evd_fast.K_conv)
                A_v_cell = compose_evd_coefficient(
                    _A_bg_field, A_v_phys, "nemo_replace",
                    convective=_evd_fast.nu_conv)
            _phys_cfg = _cfg_b.physics
            if (_phys_cfg is not None
                    and getattr(_phys_cfg.vertical_mixing, "iwm", None)
                    is not None
                    and _phys_cfg.vertical_mixing.iwm.enabled):
                # zdfiwm on the physics-provided-K FAST path (KPP pipeline
                # surfaces K_v/A_v on the tendencies): add the SAME additive
                # wave-driven contribution compute_vertical_K_profiles would
                # add on the fallback path (NEMO zdfphy order: closure first,
                # zdf_iwm adds onto avt/avm).  The non-wet-interface zeroing
                # below (_wet_if_vmix) masks it at the seafloor exactly like
                # the fallback path's tail guard.
                from legoesm.ocean.physics.vertical_mixing.k_profiles import (
                    iwm_K_profile,
                )
                from legoesm.ocean.eos import make_eos_fn as _mk_eos
                _K_iwm = iwm_K_profile(
                    state, _zc, _cfg_b.physics,
                    _phys_cfg.vertical_mixing.iwm,
                    eos_fn=_mk_eos(eos=_cfg_b.eos,
                                   eos_linear=_cfg_b.eos_linear),
                    iwm_fields=_iwm,
                ).astype(dtype)
                K_v_cell = K_v_cell + _K_iwm
                A_v_cell = A_v_cell + _K_iwm
        else:
            # Fallback: recompute K profiles (expensive for KPP).
            physics_config = _cfg_b.physics
            if physics_config is None:
                from legoesm.ocean.physics.combined import OceanPhysicsConfig
                from legoesm.ocean.physics.vertical_mixing.config import (
                    VerticalMixingConfig,
                )
                from legoesm.ocean.physics.convection.config import (
                    OceanConvectionConfig,
                )
                physics_config = OceanPhysicsConfig(
                    vertical_mixing=VerticalMixingConfig(scheme="none"),
                    convection=OceanConvectionConfig(scheme="none"),
                )
            # #1226 sh2_walk.py Candidate E/F (nemo_face_native): that TKE
            # branch of _vmix_K_profiles self-detects face-staggered vs.
            # cell-centred u/v (k_profiles.py:513) and needs the RAW
            # (uncollapsed) faces to reconstruct zdfsh2.F90's face-native
            # shear -- pre-collapsing here (as every OTHER scheme requires;
            # richardson/catke pass state.u.data straight through with no
            # staggering check of their own) would silently degrade
            # nemo_face_native to the T-collapsed Burchard geometry. Skip
            # the collapse ONLY for this scheme+option combination; every
            # other scheme/option keeps the BIT-IDENTICAL pre-collapse.
            _vmix_cfg_here = getattr(physics_config, "vertical_mixing", None)
            _keep_raw_faces = (
                _vmix_cfg_here is not None
                and _vmix_cfg_here.scheme == "tke"
                and getattr(_vmix_cfg_here.tke, "tke_shear_production",
                           "squared_centered") in ("nemo_face_native",
                                                   "nemo_face_native_now2",
                                                   "nemo_face_native_nbb2")
            )
            if _keep_raw_faces:
                cc_state = state
            else:
                u_cell = 0.5 * (state.u.data[:, :-1, :] + state.u.data[:, 1:, :])
                v_cell = 0.5 * (state.v.data[:-1, :, :] + state.v.data[1:, :, :])
                cc_state = state._replace(
                    u=state.u.replace(data=u_cell),
                    v=state.v.replace(data=v_cell),
                )
            # Use the model's own EOS for the vmix density / static-stability
            # N² so the TKE convection trigger (n2_mode="adiabatic") is
            # consistent with the dynamical core. None for Wright leaves the
            # legacy behaviour bit-identical.
            from legoesm.ocean.eos import make_eos_fn as _make_eos_fn
            _vmix_eos_fn = _make_eos_fn(
                eos=_cfg_b.eos, eos_linear=_cfg_b.eos_linear,
                eos_nemo_seos=getattr(_cfg_b, "eos_nemo_seos", None),
            )
            # PROGNOSTIC TKE carry: thread the carried TKE + the energy-recycling
            # source into the fallback K-profile solve, and capture the updated
            # TKE so the model step can store it back on the state. The TKE step
            # uses dt_mom (Veros tke.py:137). For non-prognostic configs (and
            # non-TKE schemes) ``tke_new`` comes back None ⇒ bit-identical.
            _vmix_cfg = physics_config.vertical_mixing
            # Prognostic-TKE carry fires for the Gaspar/Burchard "tke" scheme
            # (when prognostic) AND for "catke" (always prognostic).
            _tke_prognostic = (
                _vmix_cfg.scheme == "catke"
                or (_vmix_cfg.scheme == "tke"
                    and bool(getattr(_vmix_cfg.tke, "prognostic", False)))
            )
            _tke_cfg = _vmix_cfg.tke
            if (getattr(_tke_cfg, "tke_htau_evaluation", "jax_expression")
                    == "nemo_literal"):
                _tke_lat_deg = getattr(_grid, "native_lat_T_deg", None)
                if _tke_lat_deg is None:
                    raise ValueError(
                        "tke_htau_evaluation='nemo_literal' requires native "
                        "T-point degree latitudes carried by the NEMO state "
                        "bridge; grid.native_lat_T_deg is None")
            else:
                _tke_lat_deg = jnp.degrees(_grid.lat_T)
            if _tke_prognostic:
                _tke_p_sh2_result = self._tke_step_entry_p_sh2(
                    state, eta_now=eta_now, u_now=u_now, v_now=v_now,
                    z_coord=_zc, config=_cfg_b, grid=_grid,
                    return_face_metrics=return_tke_entry)
                if return_tke_entry:
                    _tke_p_sh2, _tke_shear_face_metrics_used = (
                        _tke_p_sh2_result)
                else:
                    _tke_p_sh2 = _tke_p_sh2_result
                K_v_cell, A_v_cell, tke_new = compute_vertical_K_profiles(
                    cc_state, _zc, surface_forcing, physics_config,
                    A_v_background=float(_cfg_b.A_v),
                    K_v_background=float(_cfg_b.K_v),
                    eos_fn=_vmix_eos_fn,
                    tke_old=tke_old,
                    dt_tke=(tke_rn_dt if (
                        _vmix_cfg.scheme == "tke"
                        and getattr(_vmix_cfg.tke, "tke_matrix_evaluation",
                                    "factored") == "nemo_literal"
                        and tke_rn_dt is not None) else dt_mom),
                    tke_source=tke_source, return_tke=True,
                    # T-point latitudes [deg] for the NEMO etau_htau_mode=
                    # "latitude" penetration profile (unused otherwise).
                    # 2-D lat_T (exact on the tripole, where rows curve and
                    # the legacy 1-D grid.lat is a row mean) — a 1-D (n_lat,)
                    # array cannot right-broadcast against the (n_lat, n_lon,
                    # nlev-1) columns inside nemo_etau_injection.
                    lat_deg=_tke_lat_deg,
                    iwm_fields=_iwm,
                    n2_tracers=n2_tracers,
                    tke_bottom_dirichlet=self._tke_bottom_dirichlet(state, z_coord=z_coord, config=config, grid=_grid),
                    tke_bottom_level=self._tke_bottom_level(z_coord=z_coord, config=config),
                    n2_tracers_before=n2_tracers_before,
                    tke_n2_bundle=tke_n2_bundle,
                    # NOW (Nnn) eta for the zdfevd trigger geometry
                    # (EnhancedDiffusionConfig.evd_n2_time_level=
                    # "nemo_now_before"); ignored by every other selection.
                    eta_now=eta_now,
                    tke_p_sh2=_tke_p_sh2,
                    return_tke_statement_trace=return_tke_entry,
                    tke_rhs_materialization=(
                        self._nemo_ws_test_hooks.tke_rhs_materialization),
                    tke_rhs_intermediate=(
                        self._nemo_ws_test_hooks.tke_rhs_intermediate),
                    seos_cfg=_cfg_b.eos_nemo_seos,
                )
                if (tke_new is not None
                        and hasattr(tke_new, "K_M")
                        and hasattr(tke_new, "tke_new")):
                    _tke_coeff_new = tke_new
                    _tke_entry_used = _tke_coeff_new.tke_entry
                    _tke_statement_trace_used = (
                        _tke_coeff_new.statement_trace)
                    if _tke_statement_trace_used is not None:
                        _tke_statement_trace_used = (
                            _tke_statement_trace_used._replace(
                                shear_face_metrics=(
                                    _tke_shear_face_metrics_used),
                                bn2_intermediate=(
                                    None if tke_n2_bundle is None else
                                    tke_n2_bundle.bn2_intermediate),
                                bn2_output=(
                                    None if tke_n2_bundle is None else
                                    tke_n2_bundle.rn2)))
                    tke_new = _tke_coeff_new.tke_new
                if _post_mixing:
                    # Phase 1 only (Veros set_tke_diffusivities from the
                    # carried tke[tau]): the third slot is the post-mixing
                    # CONTEXT, not an updated TKE — the budget is solved
                    # below, AFTER the tracer solve.
                    _tke_ctx, tke_new = tke_new, None
            else:
                K_v_cell, A_v_cell = compute_vertical_K_profiles(
                    cc_state, _zc, surface_forcing, physics_config,
                    A_v_background=float(_cfg_b.A_v),
                    K_v_background=float(_cfg_b.K_v),
                    eos_fn=_vmix_eos_fn,
                    lat_deg=_tke_lat_deg,
                    iwm_fields=_iwm,
                    n2_tracers=n2_tracers,
                    n2_tracers_before=n2_tracers_before,
                    eta_now=eta_now,
                    seos_cfg=_cfg_b.eos_nemo_seos,
                )

        # Private causal seam: two arrays replace the post-closure heat and
        # viscosity profiles; later arrays optionally replace formed tracer
        # K, the tracer e3w(Kmm) divisor, the tracer e3t(Kaa) matrix weight,
        # and the already-formed temperature content, in that order.
        _formed_effective_K_override = None
        _tracer_e3w_test_override = None
        _tracer_e3t_test_override = None
        _tracer_content_t_test_override = None
        _fixed_solve_test_input = None
        _fixed_solve_replace = None
        if isinstance(effective_K_test_override,
                      _NEMOVerticalSolveTestInput):
            _fixed_solve_test_input = effective_K_test_override
            _fixed_solve_replace = jnp.asarray(
                effective_K_test_override.replace, dtype=bool)
            if _fixed_solve_replace.shape != (6,):
                raise ValueError(
                    "fixed vertical-solve test input requires six selectors")
            K_v_cell = jnp.where(
                _fixed_solve_replace[0],
                effective_K_test_override.heat_K, K_v_cell)
            A_v_cell = jnp.where(
                _fixed_solve_replace[1],
                effective_K_test_override.viscosity_K, A_v_cell)
        elif effective_K_test_override is not None:
            if len(effective_K_test_override) not in (2, 3, 4, 5, 6):
                raise ValueError(
                    "vertical K test override requires 2 through 6 arrays")
            K_v_cell, A_v_cell = effective_K_test_override[:2]
            if len(effective_K_test_override) >= 3:
                _formed_effective_K_override = effective_K_test_override[2]
            if len(effective_K_test_override) >= 4:
                _tracer_e3w_test_override = effective_K_test_override[3]
            if len(effective_K_test_override) >= 5:
                _tracer_e3t_test_override = effective_K_test_override[4]
            if len(effective_K_test_override) == 6:
                _tracer_content_t_test_override = effective_K_test_override[5]

        # DIAGNOSTIC CAPTURE (return_K_profiles): the interface diffusivity
        # K_v_cell (heat, NEMO avt) and viscosity A_v_cell (momentum, avm) at
        # exactly the point the tracer/momentum solves consume them — AFTER the
        # closure, the config background floors AND the additive internal-wave
        # mixing, which is what NEMO publishes as avt/avm.  Returned before the
        # solve so the call is a pure profile read, and before the K33
        # isoneutral fold, which NEMO carries in its lateral operator, not in
        # avt.  Reads the SAME code the step runs; it is not a re-derivation.
        if return_K_profiles:
            return K_v_cell, A_v_cell

        # dz at cell centers (jacobian-corrected so the eta-stretched
        # column heights match the partial-cell / z* layer thicknesses
        # used by every other operator in this step).
        J_cell = compute_ocean_jacobian(
            state.eta.data, state.H_bathy.data, _zc,
        )
        # Diffuse on the ACTUAL per-cell thickness.  The backward-Euler solve
        # with zero-flux BCs conserves Σ(dz_cell·T) per column; for PHYSICAL heat
        # conservation that weight must be the partial-cell thickness h_partial·J,
        # not dz_ref·J — the thin bottom partial cell is NOT a full reference
        # cell, and weighting it by dz_ref leaks heat at the topography (a
        # sum(h_partial·T) drift; gated by test_partial_cells_phase7
        # ::test_partial_cells_implicit_mixing_conserves_heat).  h_partial·J ==
        # compute_layer_thickness for partial cells; below-seafloor cells get
        # h_partial=0 → dz_cell=0, which the solver clips (inv_dz via
        # maximum(dz,_EPS)) and the _wet_if_vmix / face-activity guards zero every
        # flux that would couple them, so they stay inert.  Pure z-star keeps
        # dz_ref·J → BIT-IDENTICAL (else branch == the original line).
        if isinstance(_zc, OceanPartialCellCoordinate):
            dz_cell = _zc.h_partial * J_cell[..., jnp.newaxis]
        else:
            dz_cell = _zc.dz_ref * J_cell[..., jnp.newaxis]
        # Gradient (center-to-center) divisor of the implicit solve.  Default is
        # the midpoint reconstruction 0.5(dz_k+dz_{k+1}); the Veros-faithful slot
        # (config.implicit_vmix_dzw_slot, #428) uses the coordinate's
        # center-to-center spacing dz_half_ref·J = Veros's dzw, which differs from
        # the midpoint on a u_centered z-coordinate.  NO-OP on a midpoint z-star.
        # The NEMO identity (zdf_implicit_solver_evaluation="nemo_literal", the
        # literal dyn_zdf/tra_zdf program) divides by e3w(Kmm) UNBRANCHED —
        # NEMO has no switch here, so neither do we: one canonical
        # ``nemo_e3w_kmm`` serves the tracer solve and the momentum solve
        # (e3uw_0 IS e3w_0 on the zco branch, zgr_lib.F90:111-112).  The former
        # ``implicit_vmix_e3t_now_divisor`` flag, which fixed only the time
        # level and kept the midpoint SLOT, is GONE: it selected a divisor NEMO
        # does not have, and the certified DINO card never set it (measured
        # +0.29% median / +0.90% max too large, abyss-weighted —
        # docs/ocean/fidelity/dino_zdf_divisor_scaling.md).
        # Call sites that pass a post-update AFTER state (_leapfrog_step's
        # naa_expl — the DINO kamm_mlf production path; _unsplit_ab2_step's
        # state_corr; _ab2_step's state_ab2; _step_impl's state_new) thread the
        # true NOW eta explicitly via ``eta_now``; the momentum-only friction
        # call passes the step-entry state directly, so its fallback
        # (eta_now=None -> state.eta) IS the NOW eta.  A future call site
        # that passes an AFTER state without eta_now would silently divide by
        # the AFTER thickness — thread eta_now there too, and ``u_now``/
        # ``v_now`` with it (#1455: the bottom-drag rate below has exactly the
        # same NOW-vs-AFTER problem, and is threaded from the same sites).
        # Static Python bools (feature-gating exception, CLAUDE.md) — config
        # is not traced.
        _dzw_slot = bool(getattr(_cfg_b, "implicit_vmix_dzw_slot", False))
        _zdf_literal = (getattr(
            _cfg_b, "zdf_implicit_solver_evaluation", "shared_thomas")
            == "nemo_literal")
        _zdf_legacy_w = bool(
            self._nemo_ws_test_hooks.legacy_zdf_midpoint_w_metric)
        # Cell-centered NOW thickness and NOW (1+r3t) (only computed / used
        # under the NEMO identity; both feed the u/v-face divisor below so the
        # tracer and momentum solves share one NOW-eta evaluation).
        e3t_now = None
        _stretch_now = None
        if _dzw_slot:
            dz_half_cell = (_zc.dz_half_ref
                            * J_cell[..., jnp.newaxis]).astype(dz_cell.dtype)
        elif _zdf_literal:
            _eta_now = eta_now if eta_now is not None else state.eta.data
            e3t_now = compute_layer_thickness(
                _eta_now, state.H_bathy.data, _zc,
                min_water_column_m=_cfg_b.min_water_column_m)
            _stretch_now = nemo_r3t_stretch(
                _zc, _eta_now, state.H_bathy.data)
            if _zdf_legacy_w:
                dz_half_cell = build_dz_half(e3t_now).astype(dz_cell.dtype)
            else:
                dz_half_cell = nemo_e3w_kmm(
                    _zc, e3t_now, _stretch_now).astype(dz_cell.dtype)
        else:
            dz_half_cell = build_dz_half(dz_cell)

        mask_3d = state.land_mask.data[..., jnp.newaxis]

        # The surface land mask cannot describe a partial/full-step column's
        # staircase bottom.  Keep the legacy 2-D broadcast for every existing
        # shared-Thomas card, but give the NEMO-literal application its actual
        # three-dimensional T mask.  Face masks are constructed below from the
        # same active ladder once the momentum geometry is available.
        _literal_t_wet = jnp.broadcast_to(mask_3d > 0.5, state.T.data.shape)
        if isinstance(_zc, OceanPartialCellCoordinate):
            _literal_t_wet = jnp.logical_and(
                _literal_t_wet, _zc.is_active)

        # Partial-cell dry-interface guard: no implicit flux through the
        # seafloor.  K/A at interface k couple cells k and k+1; where cell k+1
        # is below the seafloor the diffusivity must be EXACTLY zero (Veros
        # maskW), or the solve mixes the bottom wet cell with the T=S=0 (or
        # u=0) rock cells — incl. the K33 isoneutral diagonal and the A_v/K_v
        # config backgrounds, which are NOT covered by the K-profile-level
        # masking in compute_vertical_K_profiles.  Pure z-star: no-op.
        if isinstance(_zc, OceanPartialCellCoordinate):
            _wet_if_vmix = _zc.is_active.astype(
                state.T.data.dtype)[..., 1:]
        else:
            _wet_if_vmix = None

        # ---- Tracers (cell-centered: K aligns with T, S directly) ----
        T_new, S_new = state.T.data, state.S.data
        # Double-diffusion salt-heat delta — MUST be defined on every path
        # (the momentum-only friction call has do_tracers=False, yet the
        # K_s_cell construction below reads dK_ddm_salt unconditionally;
        # codex r2 UnboundLocalError fix).  None ⇒ salt uses K_v (no ddm).
        dK_ddm_salt = None
        if do_tracers:
            if return_tracer_solve_trace:
                _trace_heat_K = K_v_cell.astype(state.T.data.dtype)
                _trace_isoneutral_K = (
                    jnp.zeros_like(_trace_heat_K) if K33_iso is None
                    else K33_iso.astype(state.T.data.dtype))
            K_v_cell = K_v_cell.astype(state.T.data.dtype)
            if K33_iso is not None:
                # Fold the vertical isoneutral diffusivity K_33 into the implicit
                # tracer solve (Veros core/isoneutral/diffusion.py:154). K_33 ≥ 0 at
                # interfaces, same (n_lat, n_lon, nlev-1) shape as K_v_cell.  TRACERS
                # ONLY — momentum uses A_v_cell, which is untouched.
                K_v_cell = K_v_cell + K33_iso.astype(state.T.data.dtype)
            # Double diffusion (NEMO zdfddm): salt fingering / diffusive
            # convection add a SEPARATE heat (avt) and salt (avs) diffusivity.
            # avt folds into K_v_cell (heat) HERE — before the wet-interface
            # mask below — so the seafloor guard applies to it too; the
            # salt-heat delta ``dK_ddm_salt = avs - avt`` is carried to the
            # tracer solve so S diffuses with ``K_v + (avs - avt)``.  Momentum
            # (A_v_cell) is untouched, matching zdfddm.  Static Python gate
            # (feature-gating exception): None ⇒ bit-identical legacy pair solve.
            _ddm_cfg = getattr(
                getattr(getattr(_cfg_b, "physics", None),
                        "vertical_mixing", None), "ddm", None)
            if _ddm_cfg is not None and _ddm_cfg.enabled:
                from legoesm.ocean.physics.vertical_mixing.k_profiles import (
                    ddm_K_profile,
                )
                from legoesm.ocean.eos import make_eos_fn as _mk_eos_ddm
                _avt_ddm, _avs_ddm = ddm_K_profile(
                    state, _zc, _cfg_b.physics, _ddm_cfg,
                    eos_fn=_mk_eos_ddm(eos=_cfg_b.eos,
                                       eos_linear=_cfg_b.eos_linear),
                )
                _dt_dtype = state.T.data.dtype
                _avt_ddm = _avt_ddm.astype(_dt_dtype)
                _avs_ddm = _avs_ddm.astype(_dt_dtype)
                K_v_cell = K_v_cell + _avt_ddm
                dK_ddm_salt = _avs_ddm - _avt_ddm
            if _wet_if_vmix is not None:
                K_v_cell = K_v_cell * _wet_if_vmix
                if dK_ddm_salt is not None:
                    dK_ddm_salt = dK_ddm_salt * _wet_if_vmix
            if _fixed_solve_test_input is not None:
                K_v_cell = jnp.where(
                    _fixed_solve_replace[2],
                    _fixed_solve_test_input.formed_K, K_v_cell)
            elif _formed_effective_K_override is not None:
                K_v_cell = _formed_effective_K_override
            # IMPLICIT surface tracer forcing: add masked dt·S_surf to the
            # solve input, matching Veros's dt_tracer·forc/dz[surface] RHS.
            T_solve_in = state.T.data
            S_solve_in = state.S.data
            if surface_tracer_forcing is not None:
                _dT_surf = surface_tracer_forcing.dT_dt.data.astype(state.T.data.dtype)
                _dS_surf = surface_tracer_forcing.dS_dt.data.astype(state.S.data.dtype)
                T_solve_in = state.T.data + dt * _dT_surf * mask_3d
                S_solve_in = state.S.data + dt * _dS_surf * mask_3d
            # IMPLICIT COLUMN tracer source (sponge, EXT-N2): same weight-1.0
            # RHS-source seam as the surface forcing — Veros applies
            # tempsalt_sources to temp[taup1] BEFORE the vmix solve
            # (thermodynamics.py:419), which is algebraically this same
            # ``X_old + dt·S`` solve input.  dt is dt_tracer (Veros
            # dt_tracer·temp_source).  Kept OUT of the TKE forc_temp
            # reconstruction below (Veros: tempsalt_sources never enters
            # forc_rho_surface).  ``None`` ⇒ bit-identical.
            if tracer_source is not None:
                _dT_src = tracer_source.dT_dt.data.astype(state.T.data.dtype)
                _dS_src = tracer_source.dS_dt.data.astype(state.S.data.dtype)
                T_solve_in = T_solve_in + dt * _dT_src * mask_3d
                S_solve_in = S_solve_in + dt * _dS_src * mask_3d

        # ---- Momentum coefficient inputs (u at u-faces, v at v-faces) ----
        # Interpolate A_v and dz from cell centers to face centers.  The
        # solver only needs dz to be positive (it clips internally) and
        # the resulting tridiagonal system is well-posed on any column
        # with at least two wet levels.
        u_new, v_new = state.u.data, state.v.data
        u_solve_in, v_solve_in = state.u.data, state.v.data
        _zdf_explicit_u, _zdf_explicit_v = u_solve_in, v_solve_in
        if do_momentum and getattr(_cfg_b, "surface_stress_implicit",
                                   False) and surface_forcing is not None:
            # NEMO dynzdf surface BC: deposit the wind stress in the TOP cell
            # of the implicit solve's RHS (dt_mom * tau/(rho0 dz0)) so the
            # momentum enters TOGETHER with its vertical viscous
            # redistribution — no explicit per-step surface kick. The
            # depth-mean this adds is re-imposed to the barotropic solution
            # by nemo_stage_mean_imposition (validated at init).
            from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
                surface_stress_faces,
            )
            _sfx = surface_stress_faces(
                surface_forcing, state.u.data.dtype, _zc, J_cell,
                _grid, **({} if self._nemo_ws_test_hooks
                           .legacy_coastal_surface_stress_factors else {
                               "cell_mask": state.land_mask.data,
                               "u_mask": state.u_mask.data,
                               "v_mask": state.v_mask.data,
                           }))
            if _sfx is not None:
                _tau_i_u, _tau_j_v, _dz0u, _dz0v = _sfx
                _r0 = jnp.asarray(_cfg_b.constants.rho_0,
                                  dtype=state.u.data.dtype)
                u_solve_in = state.u.data.at[..., 0].add(
                    dt_mom * _tau_i_u / (_r0 * jnp.maximum(_dz0u, 1e-10)))
                v_solve_in = state.v.data.at[..., 0].add(
                    dt_mom * _tau_j_v / (_r0 * jnp.maximum(_dz0v, 1e-10)))
        if do_momentum:
            A_v_cell = A_v_cell.astype(state.u.data.dtype)
            if _wet_if_vmix is not None:
                A_v_cell = A_v_cell * _wet_if_vmix
            A_v_u = interp_cell_to_uface(A_v_cell)        # (n_lat, n_lon+1, nlev-1)
            dz_u = interp_cell_to_uface(dz_cell)
            # Fused v-interps: one sendrecv pair per cut for A_v + dz
            # (audit lever O4; both are independent cell fields here).
            A_v_v, dz_v = interp_to_v_points_multi((A_v_cell, dz_cell))
            # UNMASKED face thickness, kept under its own name because two
            # consumers need a POSITIVE thickness rather than the masked
            # control volume built below: the centre-to-centre gradient slot
            # ``dz_half_{u,v}`` (whose value at a closed interface is
            # irrelevant -- the viscosity there is already zero -- but which
            # must not become 1/eps).  It is NO LONGER the
            # ``zdf_drag_in_matrix`` divisor: that one is NEMO's own
            # ``e3u_3d(iku)*(1+r3u(Kaa)*umask(iku))``, built by
            # ``_nemo_dynzdf_drag_face_thickness`` at the drag block below.
            dz_u_open, dz_v_open = dz_u, dz_v
            if _wet_if_vmix is not None:
                # FACE seafloor guard (partial cells): the cell→face AVERAGE
                # leaves A_v_face = ½·A_deep at interfaces BELOW the shallower
                # neighbour's bottom (where the face is closed).  The friction
                # solve then leaks momentum across the face's seafloor into the
                # rock cells — with convective A_v ~ kappaM_max this drains the
                # wet column's transport every dt_mom, breaking ∇·(Σh·u)=0 and
                # pumping w ~100× the Veros oracle's (the 4° Antarctic-coast
                # 2Δz blowup).  Min-rule face activity (Veros maskU/maskV on
                # the W grid) zeroes those interfaces; no-op on flat bottom.
                _act_u3, _act_v3 = compute_face_masks_3d(
                    _zc.is_active, _grid)
                A_v_u = A_v_u * _act_u3.astype(A_v_u.dtype)[..., 1:]
                A_v_v = A_v_v * _act_v3.astype(A_v_v.dtype)[..., 1:]
            if _dzw_slot:
                # Veros dzw at u/v-faces (#428): dz_half_ref·J interpolated to
                # the faces with the SAME interp that built the UNMASKED face
                # thickness dz_{u,v}_open.  Deliberately NOT the masked control
                # volume: a closed interface's gradient slot is multiplied by an
                # already-zero viscosity, and masking it would make it 1/eps
                # there for no gain.
                J_u = interp_cell_to_uface(J_cell[..., jnp.newaxis])
                J_v = interp_to_v_points(J_cell[..., jnp.newaxis], _grid)
                dz_half_u = (_zc.dz_half_ref * J_u).astype(dz_u.dtype)
                dz_half_v = (_zc.dz_half_ref * J_v).astype(dz_v.dtype)
            elif _zdf_literal:
                # NEMO e3uw(Kmm)/e3vw(Kmm) at u/v-faces: the SAME canonical
                # e3w(Kmm) the tracer solve divides by, mapped to the face —
                # ``zgr_lib.F90:111-112`` sets pe3uw = pe3vw = pe3w on the zco
                # branch, so there is one object, not two.  The face map is the
                # SAME interp that built the UNMASKED dz_{u,v}_open.  NEMO
                # closes this interface with ``wumask`` (dynzdf.F90:200-203),
                # not with a zeroed e3uw, so the mask does not belong on the
                # gradient slot either.
                # DISCLOSED, not fixed here: NEMO stretches e3uw by r3u — the
                # AREA-WEIGHTED ssh average over the two T cells divided by
                # hu_0 (domqco.F90:164-167) — while this face map averages the
                # already-stretched T-point field.  That is a separate row.
                _raw_e3w0 = getattr(_zc, "nemo_e3w_0", None)
                if (not _zdf_legacy_w and _raw_e3w0 is not None
                        and jnp.asarray(_raw_e3w0).ndim == 1):
                    # OVERFLOW usrdef_zgr.F90:166-168 leaves e3uw_0/e3vw_0
                    # on the same 20 m ladder as e3w_0; qco then stretches
                    # them with r3u/r3v (domzgr_substitute.h90:132-133).
                    _h_ref_zdf = compute_layer_thickness(
                        jnp.zeros_like(_eta_now), state.H_bathy.data, _zc,
                        min_water_column_m=_cfg_b.min_water_column_m)
                    _um3, _vm3 = compute_face_masks_3d(_zc.is_active, _grid)
                    _, _, _r3u1, _r3v1 = _nemo_ws_qco_stage_faces(
                        _eta_now, _h_ref_zdf, _um3, _vm3, _grid)
                    _raw1 = jnp.asarray(_raw_e3w0)[1:]
                    dz_half_u = (_r3u1[..., None] * _raw1).astype(dz_u.dtype)
                    dz_half_v = (_r3v1[..., None] * _raw1).astype(dz_v.dtype)
                elif _zdf_legacy_w:
                    dz_half_u = build_dz_half(dz_u_open)
                    dz_half_v = build_dz_half(dz_v_open)
                else:
                    dz_half_u = nemo_e3w_kmm(
                        _zc, e3t_now, _stretch_now,
                        to_point=interp_cell_to_uface).astype(dz_u.dtype)
                    dz_half_v = nemo_e3w_kmm(
                        _zc, e3t_now, _stretch_now,
                        to_point=lambda f: interp_to_v_points(f, _grid),
                    ).astype(dz_v.dtype)
            else:
                dz_half_u = build_dz_half(dz_u_open)
                dz_half_v = build_dz_half(dz_v_open)
            if _wet_if_vmix is not None:
                # FACE CONTROL VOLUME = NEMO's ``e3u_0 * umask``.  ``dz_cell``
                # is h_partial*J, ALREADY ZERO below the seafloor, so the
                # cell->face average returns HALF a thickness at a staircase
                # face -- a face the model's own mask closes -- and that half
                # summed into the column divisor of the barotropic split.
                # NEMO builds the face thickness on the UNMASKED reference
                # ladder and applies the mask SEPARATELY when the column is
                # summed (``hu_0 = hu_0 + e3u_0*umask``, domain.F90:145), so a
                # closed face contributes exactly zero.
                #
                # THIS IS MAIN'S OWNER, PORTED VERBATIM, NOT A SECOND RULE.
                # The construction, its NEMO citations and the
                # masked-average-vs-min decision all belong to PR #1642
                # (merged to main 2026-08-22, commit 4734c2d5f), which this
                # branch predates.  An earlier revision of THIS branch fixed
                # the same defect with ``min_cell_to_uface`` and was replaced
                # by this port after review: main's rule is the more
                # NEMO-faithful one on a tilted free surface, because NEMO
                # AVERAGES the free-surface factor between the two columns
                # (``pr3u = 0.5*(e1e2t_i*ssh_i + e1e2t_{i+1}*ssh_{i+1})
                # *r1_hu_0*r1_e1e2u``, domqco.F90:219-222, the arm DINO's
                # key_qco builds) where the min rule would take the shallower
                # column's.  Scored against NEMO's own ``hu_0 + ssh_avg`` with
                # a N(0, 0.5 m) sea surface: masked average 0.0038 m mean /
                # 0.115 m max, min rule 0.280 / 1.389, pre-fix unmasked
                # average 31.7 / 370.1.  At rest on this card the two rules
                # are identical, so nothing on the DINO twin turns on the
                # choice -- but the repo must not carry two.
                dz_u = dz_u * _act_u3.astype(dz_u.dtype)
                dz_v = dz_v * _act_v3.astype(dz_v.dtype)
            u_mask_3d = state.u_mask.data[..., jnp.newaxis]
            v_mask_3d = state.v_mask.data[..., jnp.newaxis]

        # ---- #1226 NEMO dynzdf composition options (u/v ONLY) ----
        # zdf_baroclinic_only: split u_solve_in/v_solve_in into (baroclinic
        # residual, barotropic depth mean) BEFORE the solve, so the implicit
        # friction acts on the residual only (dynzdf.F90:148-150 "remove
        # barotropic velocities"); the depth mean is re-added UNCHANGED to
        # u_new/v_new after the solve (mlf_baro_corr's later re-splice is
        # already how legoESM's barotropic mode re-enters u/v elsewhere in
        # the step, so re-adding here — rather than leaving it out — is what
        # keeps this stage a no-op on the barotropic mode).  Zero-flux BCs
        # make the solve exactly conservative on the residual (Σ residual·dz
        # invariant), so round-tripping the mean is exact at A_v=0 (test 3).
        _zdf_baroclinic_only = (
            do_momentum and getattr(_cfg_b, "zdf_baroclinic_only", False))
        if _zdf_baroclinic_only:
            # OPERAND.  dynzdf.F90:150-151 subtracts the PROGNOSTIC
            # ``uu_b(ji,jj,Kaa)`` / ``vv_b(ji,jj,Kaa)`` -- the same array
            # stprk3_stg.F90:440 reads back to build zub -- not a depth mean
            # rebuilt from the solve input.  dynzdf.F90:156-159 writes the
            # bottom stress on that same uu_b(Kaa).  So take the prognostic
            # pair whenever the state carries one; a state with no barotropic
            # pair at all keeps the reconstruction, which is the only thing it
            # can do.  Same "prognostic when the state has one" rule the stage
            # correction already follows for its target.
            _uu_b_zdf = getattr(state, "uu_b", None)
            _vv_b_zdf = getattr(state, "vv_b", None)
            if _uu_b_zdf is not None and _vv_b_zdf is not None:
                _u_bt_mean = _uu_b_zdf.data[..., jnp.newaxis].astype(
                    u_solve_in.dtype)
                _v_bt_mean = _vv_b_zdf.data[..., jnp.newaxis].astype(
                    v_solve_in.dtype)
            elif _uu_b_zdf is None and _vv_b_zdf is None:
                _u_bt_mean = depth_mean(
                    u_solve_in, dz_u, _cfg_b.min_water_column_m,
                    keepdims=True)
                _v_bt_mean = depth_mean(
                    v_solve_in, dz_v, _cfg_b.min_water_column_m,
                    keepdims=True)
            else:
                raise ValueError(
                    "NEMO barotropic removal (dynzdf.F90:150-151) requires "
                    "both uu_b and vv_b, or neither")
            u_solve_in = u_solve_in - _u_bt_mean
            v_solve_in = v_solve_in - _v_bt_mean
        _zdf_baro_subtract_u, _zdf_baro_subtract_v = (
            u_solve_in, v_solve_in)

        # zdf_drag_in_matrix: NEMO's semi-implicit bottom friction goes INTO
        # the tridiagonal diagonal at each face-column's deepest wet cell
        # (dynzdf.F90:293-305) instead of the explicit RHS kick
        # (_bc_bottom_drag, disabled at the tendency stage when this flag is
        # on — see the ocean_pe_latlon_cgrid single-owner guard).  Sign:
        # NEMO's rCdU_bot <= 0 and the diagonal SUBTRACTS the SUM of the two
        # T-point rates (zwd -= zDt_2*(rCdU_bot(i+1,j)+rCdU_bot(i,j))/e3u),
        # which ADDS positive definiteness (damping); legoESM's r_eff =
        # -rCdU_bot >= 0 is the 0.5-AVERAGE of those same two rates, so
        # extra_diag = +dt_mom*r_eff/h at the bottom cell reproduces NEMO's
        # term: zDt_2 = rDt*0.5 (dynzdf.F90:97) times the SUM is rDt times the
        # AVERAGE, and dt_mom IS rDt -- so no extra factor of 2 (#1455; the
        # code and this comment both carried one until 2026-08-22).  The
        # derivation and its viscosity cross-check are at the extra_diag_u/v
        # assignment below.  This flag requires a NEMO bottom-drag scheme,
        # and under barotropic_solver="explicit_substep" it additionally
        # requires barotropic_drag_substep=True (validated at construction):
        # skipping ``_bc_bottom_drag`` here removes the barotropic
        # substep's default F_slow drag source, so the NEMO dyn_drg
        # in-subcycle drag (dynspg_ts.F90:700-706 + dyn_drg_init :1584-1642,
        # transcribed as barotropic_drag_substep) must take over.
        # Under ln_dynspg_ts (NEMO's split-explicit barotropic), NEMO ALSO
        # adds a barotropic-drag RHS correction at the bottom cell using the
        # AFTER barotropic velocity (dynzdf.F90:148-171): "add bottom stress
        # due to barotropic component only", zDt_2*(rCdU_bot sum)*uu_b(Kaa)
        # /e3u(Kaa), with rCdU_bot <= 0 so this term OPPOSES (damps) uu_b —
        # see the sign walk at the u_solve_in/v_solve_in correction below.
        # With zdf_baroclinic_only ALSO on, u_solve_in's bottom cell already
        # had the barotropic mean subtracted out; the drag correction below
        # re-applies that same damping at the bottom cell (using the SAME
        # depth mean this stage just removed), so the two flags compose
        # into NEMO's full ln_dynspg_ts treatment. Without
        # zdf_baroclinic_only, the barotropic mode is already inside
        # u_solve_in, so no separate correction is added (nothing missing).
        extra_diag_u = 0.0
        extra_diag_v = 0.0
        if do_momentum and getattr(_cfg_b, "zdf_drag_in_matrix", False):
            from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
                nemo_bottom_drag_rate_faces,
            )
            # TIME LEVEL of rCdU_bot (#1455 sibling).  NEMO computes the
            # coefficient ONCE per step in zdf_phy from uu(:,:,:,Kmm)
            # (zdfdrg.F90:174-181) and ``dyn_zdf`` only ``USE zdfdrg``
            # (dynzdf.F90:22) -- it READS the stored array at :156-159 and
            # :296 and never recomputes it (under ln_drgimp=.TRUE., DINO's
            # setting, zdf_drg_exp at :101 does not run either).  So the
            # implicit diagonal sees the SAME Kmm coefficient the barotropic
            # loop does.  ``state`` here is whatever the caller handed us,
            # which on every production path is a post-update AFTER state --
            # exactly the situation ``eta_now`` above already exists for, and
            # threaded from the same call sites.
            _u_drg = state.u.data if u_now is None else u_now
            _v_drg = state.v.data if v_now is None else v_now
            # THICKNESS row, named and deliberately NOT threaded: NEMO builds
            # the rate from e3t(...,Kmm) (zdfdrg.F90:176) while ``dz_cell``
            # here is built from ``state.eta`` -- the AFTER eta on the same
            # call sites.  It is INERT on every card that ships today: the
            # thickness reaches the rate only through nemo_loglayer's
            # Cd(h_bot) (ocean_tendency_common.py:1085-1087), and the
            # nemo_quadratic law DINO selects returns a constant ``cd = cd0``
            # (:1089-1090) without reading h_bot at all.  A card that selects
            # bottom_drag_scheme="nemo_loglayer" WOULD pick up the after-level
            # thickness -- thread a now-level dz here before shipping one.
            _r_eff_u, _r_eff_v, _is_bot_u, _is_bot_v = (
                nemo_bottom_drag_rate_faces(
                    _u_drg, _v_drg, dz_cell, _zc,
                    _cfg_b, _grid))
            _r_eff_u = _r_eff_u.astype(state.u.data.dtype)
            _r_eff_v = _r_eff_v.astype(state.v.data.dtype)
            # NEMO dynzdf.F90:293-296: zwd(iku) -= zDt_2*(rCdU_bot(i+1,j)
            # + rCdU_bot(i,j))/e3u(iku) -- a SUM of the two T-point rates
            # (no 1/2), with zDt_2 = the physical timestep (not the 2*dt
            # leapfrog form despite the name). ``_r_eff_{u,v}`` is
            # ``nemo_bottom_drag_rate_faces``'s 0.5*(...) AVERAGE of those
            # same two T-point rates.  EVERY dynzdf.F90 line number in this
            # block is from src/OCE/DYN/dynzdf.F90 (vanilla).  The DINO card
            # BUILDS cfgs/DINO/MY_SRC/dynzdf.F90, where the same statements sit
            # at :115/:174/:200/:206/:314 -- textually identical on the drag and
            # viscosity lines, so no physics differs, but do not chase these
            # numbers in the MY_SRC copy (cf. barotropic_common.py:332, which
            # warns about exactly this for its sibling file).
            # NEMO's zDt_2 = rDt*0.5 (:97) multiplies
            # the SUM, i.e. zDt_2*sum == rDt*average, and ``dt_mom`` here IS
            # rDt -- so the average is already correctly scaled and NO extra
            # factor of 2 belongs here.  The comment this replaced argued the
            # opposite and the code carried a 2x for it (#1455).
            #
            # THE DECISIVE CROSS-CHECK is the VISCOSITY in this same
            # tridiagonal, not the drag line itself.  NEMO (:182-183):
            #   zzwi = -zDt_2*(avm(i+1)+avm(i))/(e3u*e3uw) = -rDt*avm_avg/(...)
            # ours (implicit_solver.py:325):
            #   alpha = dt*A_v_avg/(dz*dz_half),  dt = dt_mom = rDt
            # Those agree EXACTLY, which is what proves dt_mom already carries
            # NEMO's half-times-sum.  The drag must not apply it a second time.
            # The zdf_baroclinic_only RHS correction below uses the identical
            # prefactor for the identical reason (dynzdf.F90:156-159).
            # Sign unchanged: raising r_eff raises the diagonal, which damps.
            #
            # THE DIVISOR (VORTEX_SMT round 11, lane round 223).  NEMO divides
            # the implicit drag term by ITS OWN U/V scale factor at the bottom
            # level, never by a two-cell average:
            #   zwd(ji,iku) = zwd(ji,iku) - zDt_2*( rCdU_bot(ji+1,jj)
            #      + rCdU_bot(ji,jj) )
            #      / (e3u_3d(ji,jj,iku)*(1._wp+r3u(ji,jj,Kaa)*umask(ji,jj,iku)))
            # (``dynzdf.f90:306``; the V twin with e3v_3d/r3v/vmask/mbkv at
            # ``:473-474``; the ln_dynspg_ts barotropic bottom-stress re-add at
            # ``:166-167`` and ``:168-169`` -- all from the compiled ppsrc of
            # the SMT-2 build VORTEX_SMT2_VEC_R8_OMIP_L1_P3).  This is the
            # IMPLICIT form: ``dynzdf.f90:120`` calls ``zdf_drg_exp`` only
            # under ``.NOT.ln_drgimp``, while these statements sit inside the
            # ``IF( ln_drgimp )`` arms opened at ``:303`` and ``:470`` and the
            # ``IF( ln_drgimp .AND. ln_dynspg_ts )`` arm opened at ``:158``,
            # which is what every card reaching this block selects.
            #
            # ``e3u_3d`` is the REFERENCE 3-D face thickness: ``domzgr.f90:186``
            # and ``:201`` read it from the mesh variable ``e3u_0``, and over z
            # partial steps that is the MIN of the two neighbouring reference T
            # thicknesses, not their average.  ``interp_cell_to_uface(dz_cell)``
            # (its own docstring: "simple average of the two cells sharing each
            # lon face") therefore overstated the seamount's bottom-cell
            # thickness by up to 134 % on 2484 of 3660 bottom U faces and
            # under-damped the deepest cell by 2.3x (round-10 receipt S5b).
            #
            # TIME LEVEL: NEMO takes the stretch factor at Kaa, the AFTER
            # level.  This divisor reads ``state.eta``, which is the SAME
            # operand ``dz_cell`` above is already built from
            # (``J_cell = compute_ocean_jacobian(state.eta.data, ...)``), so
            # THE ONE VARIABLE THIS CHANGES IS THE FACE RULE -- NEMO's
            # min-rule ``e3u_0`` with the e1e2t-weighted ``r3u`` of
            # ``domqco.F90:219-222`` -- and neither the time level nor the ssh
            # operand moves.  DISCLOSED, not claimed away: ``state.eta`` is the
            # AFTER ssh on the certified cards' stepping path, but the
            # momentum-only additive-friction call below passes the step-entry
            # state with no ``eta_now``, so on THAT lane this divisor is a Kmm
            # stretch, exactly as it was before this change (see the
            # ``eta_now`` note above).  That lane is an open row, not a
            # regression.
            #
            # No new rule is written here: ``_nemo_ws_qco_stage_faces`` is the
            # single shared assembler of ``e3u_0*(1+r3u*umask)`` that the WS-RK3
            # stage geometry and the PE lane's wzv arm already call.
            _drg_e3u, _drg_e3v = _nemo_dynzdf_drag_face_thickness(
                state.eta.data, state.H_bathy.data, _zc, _cfg_b, _grid,
                dz_u_open.dtype)
            extra_diag_u = (
                dt_mom * _r_eff_u[..., jnp.newaxis]
                / jnp.maximum(_drg_e3u, 1e-10) * _is_bot_u)
            extra_diag_v = (
                dt_mom * _r_eff_v[..., jnp.newaxis]
                / jnp.maximum(_drg_e3v, 1e-10) * _is_bot_v)
            if _zdf_baroclinic_only:
                # NEMO dynzdf.F90:156-159: puu(Krhs) += zDt_2*(rCdU_bot sum)
                # * uu_b(Kaa)/e3u(iku), with rCdU_bot <= 0 in NEMO's
                # convention -- so this RHS term is NEGATIVE (it damps the
                # barotropic bottom velocity uu_b/vv_b, opposing it, not
                # reinforcing it). legoESM's r_eff = -rCdU_bot >= 0, so the
                # sign-translated term SUBTRACTS from the solve input:
                # the barotropic-mode bottom cell loses ``dt_mom*r_eff
                # /e3u * u_bt_mean`` before the solve, matching NEMO's
                # damping direction.  Same prefactor as the diagonal above and
                # for the same reason (zDt_2*sum == rDt*average, #1455), and
                # the SAME divisor: ``dynzdf.f90:166``/``:168`` write the same
                # ``e3u_3d(iku)*(1+r3u(Kaa)*umask(iku))`` the diagonal uses.
                u_solve_in = u_solve_in - (
                    dt_mom * _r_eff_u[..., jnp.newaxis]
                    / jnp.maximum(_drg_e3u, 1e-10) * _is_bot_u * _u_bt_mean)
                v_solve_in = v_solve_in - (
                    dt_mom * _r_eff_v[..., jnp.newaxis]
                    / jnp.maximum(_drg_e3v, 1e-10) * _is_bot_v * _v_bt_mean)
        _zdf_baro_drag_u, _zdf_baro_drag_v = u_solve_in, v_solve_in

        # ---- Solve dispatch: batched (opt-in diag) / T+S pair / singles ---
        # Trace-time env switches (feature-gating exception: static
        # Python `if`, baked into the compiled graph — flip BEFORE the
        # first compile).  Field-batched solve measured a NON-win at
        # production resolution (jobs 8459136/8459145: +16%/+8% at
        # LL128 but -15%/-14% at LL192 — bandwidth-bound, batching adds
        # intermediate traffic), so it stays opt-in; it solves all four
        # fields in one call, hence only valid when BOTH gates are on.
        from legoesm.ocean.physics.vertical_mixing import (
            implicit_vertical_diffusion_ocean,
            implicit_vertical_diffusion_ocean_pair,
        )
        # #1226: the batched entry point has no extra_diag slot (it shares
        # ONE coefficient-build code path across all 4 fields); the
        # drag-in-matrix diagonal term is momentum-only, so batched mode is
        # incompatible with it — fall through to the per-field solves
        # instead of silently dropping the term (LEGOESM_VMIX_BATCHED is
        # opt-in perf-only, so this never regresses correctness).
        _vmix_batched = (
            os.environ.get("LEGOESM_VMIX_BATCHED", "0") == "1"
            and do_tracers and do_momentum
            and not getattr(_cfg_b, "zdf_drag_in_matrix", False)
        )
        # (``_zdf_literal`` — the NEMO identity — is resolved once, above the
        # divisor block that also keys off it.)
        # The source-ordered path is intentionally not routed through the
        # performance-only batched solver: doing so would replace its three
        # separate recurrences with the shared generic lowering.
        _vmix_batched = _vmix_batched and not _zdf_literal
        # Double-diffusion salinity diffusivity: K_v (heat) + (avs - avt).
        # ``dK_ddm_salt is None`` (ddm off) ⇒ K_s_cell IS K_v_cell (same
        # object) ⇒ the shared-K pair fast path stays BYTE-IDENTICAL.
        K_s_cell = K_v_cell if dK_ddm_salt is None else (K_v_cell + dK_ddm_salt)
        if _fixed_solve_test_input is not None:
            _tracer_e3w = jnp.where(
                _fixed_solve_replace[3],
                _fixed_solve_test_input.tracer_e3w, dz_half_cell)
            _tracer_e3t = jnp.where(
                _fixed_solve_replace[4],
                _fixed_solve_test_input.tracer_e3t, dz_cell)
        else:
            _tracer_e3w = (dz_half_cell
                           if _tracer_e3w_test_override is None
                           else _tracer_e3w_test_override)
            _tracer_e3t = (dz_cell
                           if _tracer_e3t_test_override is None
                           else _tracer_e3t_test_override)
        if _vmix_batched:
            T_new, S_new, u_new, v_new = (
                implicit_vertical_diffusion_ocean_batched([
                    (T_solve_in, K_v_cell, dz_cell, dz_half_cell, dt),
                    (S_solve_in, K_s_cell, dz_cell, dz_half_cell, dt),
                    (u_solve_in, A_v_u, dz_u, dz_half_u, dt_mom),
                    (v_solve_in, A_v_v, dz_v, dz_half_v, dt_mom),
                ])
            )
        else:
            if do_tracers:
                if _zdf_literal:
                    if dK_ddm_salt is not None:
                        raise ValueError(
                            "zdf_implicit_solver_evaluation='nemo_literal' "
                            "does not yet support distinct DDM heat/salt "
                            "matrices")
                    from legoesm.ocean.physics.vertical_mixing import (
                        implicit_vertical_diffusion_ocean_tracer_pair_dispatch,
                    )
                    if nemo_tracer_content_rhs is None:
                        _content_t = T_solve_in * dz_cell
                        _content_s = S_solve_in * dz_cell
                    else:
                        _content_t, _content_s = nemo_tracer_content_rhs
                    if _fixed_solve_test_input is not None:
                        _content_t = jnp.where(
                            _fixed_solve_replace[5],
                            _fixed_solve_test_input.temperature_content,
                            _content_t)
                    elif _tracer_content_t_test_override is not None:
                        _content_t = _tracer_content_t_test_override
                    _tracer_wet = _literal_t_wet
                    _tracer_result = (
                        implicit_vertical_diffusion_ocean_tracer_pair_dispatch(
                            T_solve_in, S_solve_in, _content_t, _content_s,
                            K_v_cell, _tracer_e3t, _tracer_e3w, dt,
                            _tracer_wet,
                            evaluation="nemo_literal",
                            implicit_w=nemo_aimp_tracer_w, return_matrix_trace=return_tracer_solve_trace))
                    if return_tracer_solve_trace:
                        T_new, S_new, _trace_matrix = _tracer_result
                        if (_trace_heat_K is None
                                or _trace_isoneutral_K is None):
                            raise ValueError(
                                "tracer solve trace missed its K components")
                        _trace_lower, _trace_diagonal, _trace_upper = (
                            _trace_matrix)
                        _tracer_solve_trace = _NEMOWSTracerSolveTrace(
                            heat_K=_trace_heat_K,
                            isoneutral_K=_trace_isoneutral_K,
                            effective_K=K_v_cell,
                            e3t_after=_tracer_e3t,
                            e3w_now=_tracer_e3w,
                            wet=_tracer_wet,
                            content_T=_content_t,
                            lower=_trace_lower,
                            diagonal=_trace_diagonal,
                            upper=_trace_upper,
                            solved_T=T_new,
                            viscosity_K=A_v_cell,
                        )
                    else:
                        T_new, S_new = _tracer_result
                # T and S share the IDENTICAL tridiagonal matrix (same
                # K_v_cell incl. any K33_iso fold + partial-cell wet
                # mask, same dz/dz_half/dt), so the pair solve factors
                # it ONCE — bit-identical outputs, one fewer
                # coefficient build + forward-factor sweep (vmix is
                # memory-bandwidth-bound; this REMOVES traffic where
                # field-batching ADDED it).  LEGOESM_VMIX_TSPAIR=0
                # restores the two separate solves (trace-time switch,
                # same caveat as above).
                elif (dK_ddm_salt is None
                        and os.environ.get("LEGOESM_VMIX_TSPAIR", "1") != "0"):
                    T_new, S_new = implicit_vertical_diffusion_ocean_pair(
                        T_solve_in, S_solve_in,
                        K_v_cell, dz_cell, dz_half_cell, dt,
                    )
                else:
                    # Separate T (K_v_cell) and S (K_s_cell) solves — required
                    # when double diffusion is active (avs != avt); byte-identical
                    # to the pair when dK_ddm_salt is None (K_s_cell is K_v_cell).
                    T_new = implicit_vertical_diffusion_ocean(
                        T_solve_in, K_v_cell, dz_cell, dz_half_cell, dt,
                    )
                    S_new = implicit_vertical_diffusion_ocean(
                        S_solve_in, K_s_cell, dz_cell, dz_half_cell, dt,
                    )
            if do_momentum:
                if _zdf_literal:
                    from legoesm.ocean.physics.vertical_mixing import (
                        implicit_vertical_diffusion_ocean_momentum_dispatch,
                    )
                    if isinstance(_zc, OceanPartialCellCoordinate):
                        _uwet = jnp.logical_and(
                            jnp.broadcast_to(u_mask_3d > 0.5,
                                             u_solve_in.shape),
                            _act_u3)
                        _vwet = jnp.logical_and(
                            jnp.broadcast_to(v_mask_3d > 0.5,
                                             v_solve_in.shape),
                            _act_v3)
                    else:
                        _uwet = jnp.broadcast_to(
                            u_mask_3d > 0.5, u_solve_in.shape)
                        _vwet = jnp.broadcast_to(
                            v_mask_3d > 0.5, v_solve_in.shape)
                    u_new = implicit_vertical_diffusion_ocean_momentum_dispatch(
                        u_solve_in, A_v_u, dz_u, dz_half_u, dt_mom, _uwet,
                        evaluation="nemo_literal",
                        extra_diag=extra_diag_u,
                        implicit_w=nemo_aimp_momentum_w_u)
                    v_new = implicit_vertical_diffusion_ocean_momentum_dispatch(
                        v_solve_in, A_v_v, dz_v, dz_half_v, dt_mom, _vwet,
                        evaluation="nemo_literal",
                        extra_diag=extra_diag_v,
                        implicit_w=nemo_aimp_momentum_w_v)
                else:
                    u_new = implicit_vertical_diffusion_ocean(
                        u_solve_in, A_v_u, dz_u, dz_half_u, dt_mom,
                        extra_diag=extra_diag_u,
                    )
                    v_new = implicit_vertical_diffusion_ocean(
                        v_solve_in, A_v_v, dz_v, dz_half_v, dt_mom,
                        extra_diag=extra_diag_v,
                    )
        if do_tracers:
            _tracer_apply_mask = (_literal_t_wet if _zdf_literal
                                  else mask_3d > 0.5)
            T_new = jnp.where(_tracer_apply_mask, T_new, state.T.data)
            S_new = jnp.where(_tracer_apply_mask, S_new, state.S.data)
        if do_momentum:
            if _zdf_momentum_observer is not None:
                jax.debug.callback(
                    _zdf_momentum_observer,
                    _zdf_explicit_u, _zdf_explicit_v,
                    _zdf_baro_subtract_u, _zdf_baro_subtract_v,
                    _zdf_baro_drag_u, _zdf_baro_drag_v,
                    u_new, v_new,
                    ordered=True,
                )
            if _zdf_baroclinic_only:
                # Re-add the SAME depth mean that was subtracted before the
                # solve (dynzdf.F90's barotropic component re-enters via
                # mlf_baro_corr AFTER dyn_zdf) — exactly conservative at
                # A_v=0 (test 3: strip + re-add round-trips to the input).
                u_new = u_new + _u_bt_mean
                v_new = v_new + _v_bt_mean
            _u_apply_mask = _uwet if _zdf_literal else u_mask_3d > 0.5
            _v_apply_mask = _vwet if _zdf_literal else v_mask_3d > 0.5
            u_new = jnp.where(_u_apply_mask, u_new, state.u.data)
            v_new = jnp.where(_v_apply_mask, v_new, state.v.data)
            if self._tke_realized_kdiss_active() and (
                    return_K_diss_v or (K_diss_v_w is None and do_tracers)):
                # Realized implicit-friction dissipation K_diss_v (Veros
                # friction.py:131-151): κ_f·(∂u_new/∂z)·(∂u_old/∂z) per face
                # interface with the SAME viscosity/metric the solve used,
                # averaged faces→centres (Veros ugrid_to_tgrid /
                # vgrid_to_tgrid; index-based 0.5/0.5 like the shared
                # _kediss_from_momentum_tendency mapping). No clamp (Veros
                # applies none).
                from legoesm.ocean.physics.vertical_mixing.tke import (
                    realized_implicit_friction_dissipation,
                )
                _diss_u = realized_implicit_friction_dissipation(
                    state.u.data, u_new, A_v_u, dz_half_u)
                _diss_v = realized_implicit_friction_dissipation(
                    state.v.data, v_new, A_v_v, dz_half_v)
                K_diss_v_w = (
                    0.5 * (_diss_u[:, :-1, :] + _diss_u[:, 1:, :])
                    + 0.5 * (_diss_v[:-1, :, :] + _diss_v[1:, :, :]))

        # ---- POST-MIXING TKE solve (Veros integrate_tke placement) ----
        # Runs AFTER the implicit tracer solve, in the call that owns the
        # tracers (the authoritative TKE site): recompute the SIGNED
        # adiabatic N² from the MIXED T/S over the dzw slot (Veros
        # calc_eq_of_state(taup1) → Nsqr[taup1]), build the surface
        # buoyancy-flux P_diss_v slot from the post-mixing surface T/S +
        # the implicit surface forcing (surf_densityf → diag_P_diss_v),
        # and run ONE backward-Euler TKE step with dt = dt_mom.
        if _post_mixing and do_tracers and _tke_ctx is not None:
            from legoesm.ocean.eos import (
                compute_buoyancy_frequency_adiabatic,
            )
            from legoesm.ocean.physics.vertical_mixing.tke import (
                compute_surface_buoyancy_P_diss_v, tke_integrate_post_mixing,
            )
            tke_cfg = _cfg_b.physics.vertical_mixing.tke
            T_n2, S_n2 = T_new, S_new
            if isinstance(_zc, OceanPartialCellCoordinate):
                # Same sub-seafloor guard the phase-1 N² used (k_profiles):
                # extend the deepest ACTIVE T/S downward so the seafloor
                # interface reads neutral, not the T=S=0 rock fill.
                from legoesm.ocean.vertical import extrapolate_below_seafloor
                T_n2 = extrapolate_below_seafloor(T_n2, _zc)
                S_n2 = extrapolate_below_seafloor(S_n2, _zc)
            # Veros recomputes Nsqr[taup1] at the STATIC reference pressures
            # (press = abs(zt)); reuse the phase-1 cell pressures (the
            # hydrostatic p of the pre-solve state — the mixing step does
            # not move the pressure field).
            N2_post = compute_buoyancy_frequency_adiabatic(
                T_n2, S_n2, _tke_ctx.p_cell,
                _zc.dz_ref, J_cell,
                eos_fn=_tke_ctx.eos_fn, rho_ref=_tke_ctx.rho_0,
                g=_tke_ctx.g, dz_half=_tke_ctx.dz_half,
            )
            # Surface kinematic fluxes [K·m/s] (Veros forc_temp_surface =
            # dzt·rate; thermodynamics.py:276), reconstructed as the COLUMN
            # SUM of the implicit forcing RATE × the live thickness.  For
            # surface-only rates this is bit-identical to rate[...,0]·dz_top
            # (the deeper terms add exact zeros).  When the rate carries a
            # penetrative-solar column (flux_feedback q_solar), the sum
            # telescopes back to the solar-INCLUSIVE total — exactly Veros's
            # forc_temp_surface, whose qnet includes the full qsol while the
            # pen(0)=0 temp_source redistribution stays OUT of
            # forc_rho_surface (top-cell rate alone would be short by
            # qsol·I(z₁)).  Exact on full-depth columns; shallow columns
            # differ only by the below-kbot leak (I(z_kbot)·qsol) — in the
            # TKE buoyancy-flux CLOSURE TERM only; the temperature tendency
            # itself stays Veros-faithful.  The leak is small for kbot deeper
            # than ~30-50 m (~0.5% of qsol at 100 m) but reaches ~28% of qsol
            # for the shallowest min_depth≈10 m single-cell shelf columns
            # (review-quantified; second-order, accepted).
            # CAVEAT (EXT-N2, realized): a column source that is NOT a
            # surface flux must NOT ride surface_tracer_forcing through this
            # sum — Veros keeps tempsalt_sources out of forc_rho_surface.
            # The implicit SPONGE rates therefore arrive on the SEPARATE
            # ``tracer_source`` argument, which this reconstruction
            # deliberately ignores.
            _sfc_T = T_new[..., 0]
            _sfc_S = S_new[..., 0]
            if surface_tracer_forcing is not None:
                _forc_T = (jnp.sum(
                    surface_tracer_forcing.dT_dt.data * dz_cell, axis=-1)
                    * state.land_mask.data)
                _forc_S = (jnp.sum(
                    surface_tracer_forcing.dS_dt.data * dz_cell, axis=-1)
                    * state.land_mask.data)
            else:
                _forc_T = jnp.zeros_like(_sfc_T)
                _forc_S = jnp.zeros_like(_sfc_S)
            P_diss_v_sfc = compute_surface_buoyancy_P_diss_v(
                _sfc_T, _sfc_S, _tke_ctx.p_cell[..., 0],
                _forc_T, _forc_S, _tke_ctx.eos_fn,
                rho_0=_tke_ctx.rho_0, g=_tke_ctx.g,
            ) * state.land_mask.data
            if self._tke_realized_kdiss_active():
                if K_diss_v_w is None:
                    raise ValueError(
                        "shear_production='realized_veros' needs the realized "
                        "K_diss_v: solve momentum in this call "
                        "(do_momentum=True) or pass K_diss_v_w from the "
                        "momentum-only call (return_K_diss_v=True).")
                _kdiss = K_diss_v_w
            else:
                _kdiss = _tke_ctx.K_M_old * _tke_ctx.shear_sq
            tke_new = tke_integrate_post_mixing(
                _tke_ctx, N2_post, _kdiss, P_diss_v_sfc,
                dt=dt_mom, cfg=tke_cfg, external_source=tke_source,
            )
            if _wet_if_vmix is not None:
                tke_new = tke_new * _wet_if_vmix

        state_out = state._replace(
            u=state.u.replace(data=u_new),
            v=state.v.replace(data=v_new),
            T=state.T.replace(data=T_new),
            S=state.S.replace(data=S_new),
        )
        if _tke_coeff_new is not None:
            from legoesm.core.field import Field
            state_out = state_out._replace(
                tke_avm=Field(data=_tke_coeff_new.K_M, name="tke_avm",
                              dims=("lat", "lon", "level"), units="m^2/s"),
                tke_avt=Field(data=_tke_coeff_new.K_H, name="tke_avt",
                              dims=("lat", "lon", "level"), units="m^2/s"),
                tke_avm_surface=(
                    None if _tke_coeff_new.K_M_surface is None else
                    Field(data=_tke_coeff_new.K_M_surface,
                          name="tke_avm_surface", dims=("lat", "lon"),
                          units="m^2/s")),
                tke_dissl=(
                    None if _tke_coeff_new.dissl is None else
                    Field(data=_tke_coeff_new.dissl, name="tke_dissl",
                          dims=("lat", "lon", "level"), units="s^-1")),
            )
        if return_K_diss_v:
            return state_out, K_diss_v_w
        if return_tke:
            if return_tke_entry:
                if _tke_entry_used is None:
                    raise ValueError(
                        "WRITE-only TKE entry trace requested, but the "
                        "closure did not expose its consumed energy operand")
                if _tke_statement_trace_used is None:
                    raise ValueError(
                        "WRITE-only TKE statement trace requested, but the "
                        "closure did not expose its production boundaries")
                return (state_out, tke_new, _tke_entry_used,
                        _tke_statement_trace_used)
            if return_tracer_solve_trace:
                if _tracer_solve_trace is None:
                    raise ValueError(
                        "WRITE-only tracer solve trace was requested but the "
                        "NEMO-literal tracer solve did not expose it")
                return state_out, tke_new, _tracer_solve_trace
            return state_out, tke_new
        if return_tracer_solve_trace:
            if _tracer_solve_trace is None:
                raise ValueError(
                    "WRITE-only tracer solve trace was requested but the "
                    "NEMO-literal tracer solve did not expose it")
            return state_out, _tracer_solve_trace
        return state_out

    def diagnose_vertical_K(self, state: LatLonCGridOceanState, dt: float,
                            surface_forcing=None, *, grid=None):
        """The vertical diffusivity and viscosity the step consumes, avt/avm.

        NEMO publishes ``avm`` and ``avt`` in its five-day output; ours are
        built inside the implicit vertical solve. This runs the
        SAME setup the step runs — ``_step_impl`` with the implicit mixing
        turned off to obtain the surfaced ``K_v_phys``/``tke_source`` inputs,
        then ``_apply_implicit_vertical_mixing`` on that explicit state with
        ``return_K_profiles`` —
        and returns the coefficients at the point the solve reads them, AFTER
        the closure, the config background and the additive internal-wave
        mixing. It is the model's own code, not a re-derivation, so the number
        is directly comparable against NEMO's published field.

        Returns ``(K_H, K_M)`` at interior interfaces.  The private compiled-
        bn2 hook also returns the exact ``rn2/rn2b`` bundle consumed here.
        """
        _grid = grid if grid is not None else self.grid
        _tke_n2_bundle = self._tke_step_entry_n2_bundle(state)
        state_expl, (K_v_phys, A_v_phys, k33_implicit,
                     surface_tracer_forcing, tke_source,
                     diss_incr, tracer_source) = self._step_impl(
            state, dt, surface_forcing=surface_forcing,
            _apply_implicit_vmix=False, grid=_grid,
            _tke_n2_bundle_override=_tke_n2_bundle)
        _tke_prog = self._tke_prognostic_active()
        _tke_old = (state.tke.data if (_tke_prog and state.tke is not None)
                    else None)
        dt_mom = dt / self.config.dt_mom_ratio
        # Mirror the step's tracer solve exactly, but stop at the profile: same
        # N² (step-entry, before-advection), same carried TKE, same NOW eta,
        # same surfaced physics K.
        profiles = self._apply_implicit_vertical_mixing(
            state_expl, dt, surface_forcing,
            K_v_phys=K_v_phys, A_v_phys=A_v_phys,
            dt_mom=dt_mom, tke_old=_tke_old, tke_source=tke_source,
            n2_tracers=self._n2_before_advection_tracers(state),
            n2_tracers_before=self._n2_nemo_before_tracers(state),
            tke_n2_bundle=_tke_n2_bundle,
            eta_now=state.eta.data,
            u_now=state.u.data, v_now=state.v.data,
            return_K_profiles=True, grid=_grid,
        )
        if self._nemo_ws_test_hooks.bn2_intermediate:
            return (*profiles, _tke_n2_bundle.rn2, _tke_n2_bundle.rn2b)
        return profiles

    def step(self, state: LatLonCGridOceanState, dt: float,
             freshwater=None, surface_forcing=None,
             sponge=None, *, grid=None,
             vertex_mask=None, t_seconds=None,
             external_tracer_rate=None,
             _shortwave_tendency_test_delta=None,
             _vertical_K_test_override=None,
             _nemo_stage1_zad_eta_after_override=None,
             ) -> LatLonCGridOceanState:
        """Advance one time step using split-explicit stepping.

        ``external_tracer_rate`` (optional ``(dT_dt, dS_dt)`` array pair,
        full-column, masked, units degC/s and PSU/s): summed into the
        EXPLICIT tracer RHS on the leap-frog-family Nnn advective pass ONLY
        (#1492 DINO ``surface_tendency_placement="leapfrog_rhs"`` — see
        ``DINOConfig`` docstring + ``_step_impl``'s ``_external_tracer_rate``).
        Requires ``outer_integrator`` in ``("leapfrog", "nemo_mlf")`` (P2:
        ``_nemo_mlf_step`` threads it to the SAME kwarg); ``None`` (default,
        every other outer integrator) ⇒ bit-identical, no-op elsewhere.

        Eager Python shim over the JIT-compiled ``_step_jitted``: fills
        the build-once vertex-mask cache from CONCRETE state BEFORE the
        body traces (an in-jit fill is impossible — the state is a
        tracer there), so the tendencies capture the mask as a closure
        constant instead of re-deriving it (with its N-S halo exchange)
        every step (census job 8474554).  For use inside an outer JIT
        context (e.g. ``lax.scan``), call ``_step_impl`` directly to
        avoid nested JIT boundaries (the in-graph vertex-mask fallback
        then applies — correct, just without the constant-fold win).

        Parameters
        ----------
        state : LatLonCGridOceanState
        dt : float
            Time step [seconds].
        freshwater : FreshwaterForcing or None
            Freshwater forcing (P, E, runoff, ice).  If None, no
            freshwater mass/salt flux is applied.
        surface_forcing : OceanSurfaceForcing or None

        ``grid``/``vertex_mask`` (optional, SPMD): default ``None`` →
        the model's own ``self.grid``/``self._vertex_mask`` (bit-identical
        single-device path); a future ``shard_map`` wrapper passes a
        per-device band-local grid (+ vertex mask) — threaded into the
        jitted body and every sub-call that reads ``self.grid``.

        Returns
        -------
        LatLonCGridOceanState
        """
        # A direct eager step is also a valid cold-start driver.  Seed the
        # NEMO pre-tke_avn coefficient memory here as well as in
        # seed_scan_carry(); otherwise only scan/restart drivers get the
        # faithful avm_k/avt_k lifetime and a bare step fails on its first
        # closure call.
        state = self._seed_tke_preclosure_carry(state)
        if self.config.barotropic.barotropic_solver == "rigid_lid":
            # Eager fail-fast (host-side, before the jitted body): the rigid-lid
            # streamfunction solve is single-rank only.  The in-body guard runs
            # at TRACE time only, so without this eager check a distributed run
            # could reuse a serially-traced compiled _step_jitted.  Cheap host
            # predicate; also covers integrate()/step_checked (both call step)
            # and integrate_scan's jax.eval_shape(self.step, ...) probe.
            from legoesm.ocean.dynamics.rigid_lid_latlon_cgrid import (
                assert_rigid_lid_single_rank,
            )
            assert_rigid_lid_single_rank()
        # Warm BOTH build-once caches (vertex mask + — rigid lid only — the
        # host-side island/streamfunction decomposition) from the CONCRETE input
        # state before crossing the jit boundary, so the jitted body reuses them
        # as compile-time constants.  Without the rigid-lid warm the first
        # rigid_lid step would try to build the numpy flood-fill from the TRACED
        # mid-step state inside _step_jitted (np.asarray on a tracer raises).
        # Single warming entry point shared with external _step_impl drivers.
        self.prime_step_caches(state)
        # Eager fail-fast (dispatch hardening): an ENABLED equilibrium tide
        # with no model time supplied would otherwise be a SILENT no-op (the
        # in-body gate is ``enabled and t_seconds is not None``) — a driver
        # stepping via bare ``step(state, dt)`` would run a tide-free
        # simulation while the config says tides are on.  integrate()/
        # integrate_scan() thread t automatically; direct steppers must pass
        # ``t_seconds`` (elapsed model seconds; t=0 = constituent epoch).
        _tf = getattr(self.config, "tidal_forcing", None)
        if _tf is not None and _tf.enabled and t_seconds is None:
            raise ValueError(
                "tidal_forcing.enabled=True but step() was called without "
                "t_seconds — the equilibrium tide needs the elapsed model "
                "time and would otherwise be SILENTLY inert.  Pass "
                "t_seconds=<elapsed seconds device scalar> to step()/"
                "step_checked(), or drive the run via integrate()/"
                "integrate_scan() which thread it automatically."
            )
        if (external_tracer_rate is not None
                and getattr(self.config, "outer_integrator", "forward_euler")
                not in ("leapfrog", "nemo_mlf")):
            raise ValueError(
                "external_tracer_rate is only consumed by the leap-frog-"
                "family Nnn advective pass (config.outer_integrator="
                "'leapfrog' or 'nemo_mlf' -- _nemo_mlf_step threads it to "
                "the SAME _external_tracer_rate kwarg, per nemo_mlf_step_"
                "transcription_spec.md §2 'already conformant' row 24); "
                f"got outer_integrator={self.config.outer_integrator!r}. "
                "Passing it under another integrator would be a silent "
                "no-op.")
        if ((_shortwave_tendency_test_delta is not None
             or _vertical_K_test_override is not None
             or _nemo_stage1_zad_eta_after_override is not None)
                and self.config.outer_integrator != "forward_euler"):
            raise ValueError(
                "private GYRE causal-arm overrides are implemented only for "
                "outer_integrator='forward_euler'.")
        # ``step`` is the production-compiled entry point even when a caller
        # has enabled JAX's process-wide diagnostic ``disable_jit`` context.
        # Inheriting that context used to bypass this boundary and execute a
        # different eager arithmetic graph (most visibly at the Roquet/HPG
        # boundary), so an eager-vs-production comparison was not comparing
        # the same program.  Re-enable only this explicitly jitted kernel;
        # private hooks remain inside the identical compiled step.
        with jax.disable_jit(False):
            result = self._step_jitted(
                state, dt, freshwater, surface_forcing, sponge,
                grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds,
                external_tracer_rate=external_tracer_rate,
                _shortwave_tendency_test_delta=_shortwave_tendency_test_delta,
                _vertical_K_test_override=_vertical_K_test_override,
                _nemo_stage1_zad_eta_after_override=(
                    _nemo_stage1_zad_eta_after_override))
            if self._nemo_ws_test_hooks.expose_live_stage_operands:
                # Returning the diagnostic tuple changes XLA's optimization
                # boundary and can move a last-bit rounding in the prognostic
                # result.  Keep the trace compiled, but source ``state_after``
                # from an independently compiled ordinary forward-Euler call.
                # The round-51 observer control requires that returned state to
                # equal the production call bit for bit.
                state_after = self._step_live_operand_reference_jitted(
                    state, dt, freshwater, surface_forcing, sponge,
                    grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds,
                    _shortwave_tendency_test_delta=(
                        _shortwave_tendency_test_delta),
                    _vertical_K_test_override=_vertical_K_test_override,
                    _nemo_stage1_zad_eta_after_override=(
                        _nemo_stage1_zad_eta_after_override))
                return result._replace(state_after=state_after)
            if (self._nemo_ws_test_hooks.tracer_process_trace is not None
                    or self._nemo_ws_test_hooks.tracer_ldf_diagnostics is not None):
                # As with the live stage operands, returning extra arrays can
                # change XLA fusion.  The next-step state therefore comes only
                # from an independently compiled ordinary production call.
                state_after = self._step_live_operand_reference_jitted(
                    state, dt, freshwater, surface_forcing, sponge,
                    grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds,
                    _shortwave_tendency_test_delta=(
                        _shortwave_tendency_test_delta),
                    _vertical_K_test_override=_vertical_K_test_override,
                    _nemo_stage1_zad_eta_after_override=(
                        _nemo_stage1_zad_eta_after_override))
                return result._replace(state_after=state_after)
            return result

    @partial(jax.jit, static_argnums=(0,))
    def _step_live_operand_reference_jitted(
            self, state: LatLonCGridOceanState, dt: float,
            freshwater=None, surface_forcing=None, sponge=None, *, grid=None,
            vertex_mask=None, t_seconds=None,
            _shortwave_tendency_test_delta=None,
            _vertical_K_test_override=None,
            _nemo_stage1_zad_eta_after_override=None,
            ) -> LatLonCGridOceanState:
        """Ordinary compiled result paired with the private operand trace."""
        new_state = self._step_impl(
            state, dt, freshwater=freshwater,
            surface_forcing=surface_forcing, sponge=sponge,
            grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds,
            _shortwave_tendency_test_delta=_shortwave_tendency_test_delta,
            _vertical_K_test_override=_vertical_K_test_override,
            _nemo_stage1_zad_eta_after_override=(
                _nemo_stage1_zad_eta_after_override))
        if self.config.polar_filter.use_polar_filter:
            new_state = self._apply_polar_filter(new_state, dt, grid=grid)
        if self.config.freeze_floor:
            new_state = self._apply_freeze_floor(new_state)
        if getattr(self.config, "ew_cyclic_overlap", False):
            new_state = self._apply_ew_cyclic_overlap(new_state)
        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def _step_jitted(self, state: LatLonCGridOceanState, dt: float,
                     freshwater=None, surface_forcing=None,
                     sponge=None, *, grid=None,
                     vertex_mask=None, t_seconds=None,
                     external_tracer_rate=None,
                     _shortwave_tendency_test_delta=None,
                     _vertical_K_test_override=None,
                     _nemo_stage1_zad_eta_after_override=None,
                     ) -> LatLonCGridOceanState:
        """JIT body of :meth:`step` (split out so the vertex-mask cache
        fill runs eagerly — see the ``step`` docstring).

        ``self`` is the only static argument; ``grid``/``vertex_mask``
        are normal (traced-capable) arguments with a ``None`` default
        (resolved to ``self.grid``/``self._vertex_mask`` ⇒ bit-identical
        single-device path) and are threaded into ``_ab2_step`` /
        ``_step_impl`` / ``_apply_polar_filter``."""
        _oi = getattr(self.config, "outer_integrator", "forward_euler")
        _valid_oi_jitted = ("forward_euler", "ab2", "leapfrog", "nemo_mlf")
        if _oi not in _valid_oi_jitted:
            raise ValueError(
                f"config.outer_integrator must be one of {_valid_oi_jitted}, "
                f"got {_oi!r}")
        if self._nemo_ws_test_hooks.expose_barotropic_substeps:
            if _oi != "forward_euler":
                raise ValueError(
                    "expose_barotropic_substeps is a private forward_euler "
                    "WS-RK3 fidelity hook")
            return self._step_impl(
                state, dt, freshwater=freshwater,
                surface_forcing=surface_forcing, sponge=sponge,
                grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds,
                _return_barotropic_substeps=True,
                _nemo_stage1_zad_eta_after_override=(
                    _nemo_stage1_zad_eta_after_override))
        if self._nemo_ws_test_hooks.expose_live_stage_operands:
            if _oi != "forward_euler":
                raise ValueError(
                    "expose_live_stage_operands is a private forward_euler "
                    "WS-RK3 fidelity hook")
            return self._step_impl(
                state, dt, freshwater=freshwater,
                surface_forcing=surface_forcing, sponge=sponge,
                grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds,
                _return_live_stage_operands=True,
                _nemo_stage1_zad_eta_after_override=(
                    _nemo_stage1_zad_eta_after_override))
        if self._nemo_ws_test_hooks.tracer_process_trace is not None:
            if _oi != "forward_euler":
                raise ValueError(
                    "tracer_process_trace is a private forward_euler "
                    "WS-RK3 fidelity hook")
            return self._step_impl(
                state, dt, freshwater=freshwater,
                surface_forcing=surface_forcing, sponge=sponge,
                grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds,
                _return_tracer_process_trace=True,
                _vertical_K_test_override=_vertical_K_test_override,
                _nemo_stage1_zad_eta_after_override=(
                    _nemo_stage1_zad_eta_after_override))
        if self._nemo_ws_test_hooks.tracer_ldf_diagnostics is not None:
            if _oi != "forward_euler":
                raise ValueError(
                    "tracer_ldf_diagnostics is a private forward_euler "
                    "WS-RK3 fidelity hook")
            return self._step_impl(
                state, dt, freshwater=freshwater,
                surface_forcing=surface_forcing, sponge=sponge,
                grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds,
                _return_ldf_diagnostic_trace=True,
                _nemo_stage1_zad_eta_after_override=(
                    _nemo_stage1_zad_eta_after_override))
        if self.config.barotropic.barotropic_solver == "implicit_unsplit":
            # MITgcm-faithful UNSPLIT implicit free surface (no barotropic/baroclinic
            # mode split). One AB2 predictor on the FULL 3D velocity + one implicit
            # elliptic eta solve + uniform surface-pressure correction. Removes the
            # split's grid-scale PGF/continuity adjointness violation that drives the
            # spurious 2dx baroclinic instability (docs/ocean/fidelity/
            # mitgcm_unsplit_freesurface_fix.md). Requires the ab2 outer integrator.
            if _oi != "ab2":
                raise ValueError(
                    "barotropic_solver='implicit_unsplit' requires "
                    "outer_integrator='ab2'.")
            # (tidal_forcing + implicit_unsplit is rejected at construction in
            # _validate_config — the unsplit solver has no barotropic-substep
            # forcing hook.)
            new_state = self._unsplit_ab2_step(
                state, dt, freshwater=freshwater,
                surface_forcing=surface_forcing, sponge=sponge,
                grid=grid, vertex_mask=vertex_mask)
        elif _oi == "ab2":
            if self.config.tracer_time_integrator == "ab2":
                raise ValueError(
                    "outer_integrator='ab2' double-counts with "
                    "tracer_time_integrator='ab2'; set the inner one to 'euler'.")
            # The faithful AB2 path (``_ab2_step``) AB2-extrapolates ONLY the
            # explicit tendency and applies implicit vertical mixing ONCE
            # afterward (Veros core/thermodynamics.py + core/external/
            # solve_stream.py), so it is unconditionally stable in the vertical
            # and compatible with convective adjustment — no convection guard.
            new_state = self._ab2_step(
                state, dt, freshwater=freshwater,
                surface_forcing=surface_forcing, sponge=sponge,
                grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds)
        elif _oi == "leapfrog":
            new_state = self._leapfrog_step(
                state, dt, freshwater=freshwater,
                surface_forcing=surface_forcing, sponge=sponge,
                grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds,
                external_tracer_rate=external_tracer_rate)
        elif _oi == "nemo_mlf":
            # P2 (docs/ocean/fidelity/nemo_mlf_step_transcription_spec.md §4/§7):
            # wire the P1 single-pass transcription (stpmlf.F90) behind its own
            # outer_integrator value. "leapfrog" (the two-pass _leapfrog_step)
            # is UNCHANGED and stays the default for every existing card -- this
            # is a NEW, separate branch, not a replacement.
            new_state = self._nemo_mlf_step(
                state, dt, freshwater=freshwater,
                surface_forcing=surface_forcing, sponge=sponge,
                grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds,
                external_tracer_rate=external_tracer_rate)
        else:
            new_state = self._step_impl(state, dt, freshwater=freshwater,
                                        surface_forcing=surface_forcing,
                                        sponge=sponge,
                                        grid=grid, vertex_mask=vertex_mask,
                                        t_seconds=t_seconds,
                                        _shortwave_tendency_test_delta=(
                                            _shortwave_tendency_test_delta),
                                        _vertical_K_test_override=(
                                            _vertical_K_test_override),
                                        _nemo_stage1_zad_eta_after_override=(
                                            _nemo_stage1_zad_eta_after_override))
        # Feature-gated on a STATIC config bool (CLAUDE.md feature-gating
        # exception): a Python ``if`` selects the branch at trace time, so
        # the freeze-floor clamp is only traced when enabled — no jnp.where
        # double-trace, bit-exact for legacy configs.
        # Fourier polar filter (STATIC config-bool gate): truncate the zonal modes
        # that exceed the per-latitude CFL near the converging-meridian poles.
        # After the prognostic update so it damps whatever the step produced, and
        # BEFORE the freeze floor so the surface temperature cap is the final word
        # (the zonal filter can otherwise pull a surface cell back below freezing).
        if self.config.polar_filter.use_polar_filter:
            new_state = self._apply_polar_filter(new_state, dt, grid=grid)
        if self.config.freeze_floor:
            new_state = self._apply_freeze_floor(new_state)
        # ORCA east-west cyclic-overlap (tripole seam) — LAST, so the halo
        # columns exactly mirror their overlap partners after every other
        # post-step projection (static config-bool gate; default off).
        if getattr(self.config, "ew_cyclic_overlap", False):
            new_state = self._apply_ew_cyclic_overlap(new_state)
        return new_state

    def _carry_nemo_rk3_after_ssh(self, entry_state, new_state):
        """Leave NEMO's next-step after-SSH guess in the state it hands on.

        At the end of every RK3 step, after the Nbb<==>Naa rotation
        (stprk3.F90:221), NEMO assigns
        ``ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)`` (stprk3.F90:225),
        where Nbb now holds the height this step PRODUCED and Naa still holds
        the height it ENTERED with (Nbb is untouched across the three stages).
        The next step reads that slot as ``r3t(:,:,Kaa)`` (stp2d.F90:149)
        immediately before its first ``CALL wzv`` (stp2d.F90:153), and NEMO
        carries it across a restart as ``ssha`` (restart.F90:184).

        ``2*a`` is exact in binary floating point, so the subtraction is the
        only rounding here and it is the one NEMO performs.  Inert unless the
        card STATES the carried form -- the same predicate the ``wzv`` call
        reads the slot with, so the writer and the reader cannot disagree.
        """
        if not nemo_rk3_after_ssh_is_carried(self.config):
            return new_state
        return new_state._replace(
            eta_rk3_after=new_state.eta.replace(
                data=2.0 * new_state.eta.data - entry_state.eta.data))

    def _apply_polar_filter(self, state: LatLonCGridOceanState, dt: float,
                            *, grid=None) -> LatLonCGridOceanState:
        """Apply the mask-aware Fourier polar filter to the prognostic fields.

        A global lat-lon ocean has ``dx = R*dlon*cos(lat) -> 0`` at the poles, so
        explicit advection/metric terms violate CFL poleward and the cold-start
        blows up (~day 0.25) regardless of the time integrator.  The shared
        ``grids.polar_filter`` (already used by the atmosphere C-grid) truncates,
        at each latitude poleward of ``polar_filter_cutoff_lat_deg``, the zonal
        Fourier wavenumbers above the CFL cap ``k_max = sf*sqrt(3)*R*cos(lat)/
        (c_max*dt)``.

        MASK-AWARE: a naive zonal FFT over a basin with continents would smear
        land zeros into adjacent ocean and couple basins across land.  So for each
        field, land cells are first filled with the per-latitude (per-level) OCEAN
        zonal mean, the filter is applied, and land cells are then restored to
        their original value.

        CONSERVATION: the ``k=0`` (zonal-mean) mode is never truncated, so the
        filter preserves the zonal mean of the *filled* field; but discarding the
        filtered values that landed on land positions perturbs the OCEAN-only zonal
        mean, so we add back the per-latitude ocean-mean deficit, restoring the
        exact per-latitude WET-CELL zonal MEAN.  Because cell area is constant in
        longitude within a row, for ``eta`` this conserves the per-row ocean volume
        exactly, and for ``u`` it preserves the zonal-mean flow (no spurious net
        zonal current).  For ``T``/``S`` it conserves the per-row, per-level
        unweighted wet-cell mean, which equals the area·thickness-weighted tracer
        content EXACTLY only where the layer thickness is zonally uniform (full
        cells, rigid lid); under partial cells / z* ``eta`` the per-row content is
        conserved only approximately (residual bounded by the partial-cell/z*
        thickness fraction).  This is a stability filter, not a flux operator, and
        runs after the conservation fixer; the residual is small and acceptable.

        Vertical masking: 2D ``land_mask``/``u_mask``/``v_mask`` are used at every
        level (not the partial-cell 3D wet mask).  Below-seafloor slots are inert
        in the dynamics, so filtering them is harmless for the prognostic update;
        if a future diagnostic reads those slots, switch to ``compute_face_masks_3d``.

        Dynamical caveat: filtering prognostic ``u``/``v``/``eta`` every step is a
        standard polar-filter practice but the latitude-dependent zonal filter does
        not commute with meridional differencing, so it can perturb discrete
        geostrophic balance poleward of the cutoff.  This is validated by the
        cold-start run + a visual/noise check, not asserted a priori.

        Staggering:
          * ``T``, ``S`` cell-centered (n_lat, n_lon, nlev) -> cell mask, land_mask
          * ``v`` v-face       (n_lat+1, n_lon, nlev)        -> v-face mask, v_mask
          * ``u`` u-face       (n_lat, n_lon+1, nlev)        -> cell mask on the
            periodic interior ``u[:, :-1]`` (last col == first), then re-wrapped
          * ``eta`` 2D cell-centered (n_lat, n_lon)          -> cell mask, land_mask

        LIMITATION: even mask-aware, a lat-lon grid cannot be fully faithful in
        the land-locked Arctic (residual cross-pole basin coupling); the faithful
        OMIP path uses the ORCA tripole.  This keeps the lat-lon grid STABLE and
        good for a tropics/mid-lat/SH comparison.
        """
        cfg = self.config
        # SPMD: default ``grid=None`` → ``self.grid`` (bit-identical
        # single-device path); a band-local grid is injected by a future
        # ``shard_map`` wrapper.
        _grid = grid if grid is not None else self.grid
        # The model's ``self.grid`` is a ``LatLonCGridGeometry`` (per-stagger
        # metrics); it carries the 1D ``lat``/``cos_lat``/``dlat`` but NOT the
        # v-face ``lat_v``/``cos_lat_v`` that ``compute_polar_filter_mask`` reads.
        # Build a minimal duck-typed grid for the filter from the geometry's own
        # arrays (so the mask latitudes match the actual model grid), deriving
        # the v-face coords with the shared ``compute_v_face_coords``.
        from types import SimpleNamespace
        dlat = float(_grid.dlat)
        if dlat == 0.0:
            raise ValueError(
                "use_polar_filter requires a regular lat-lon grid (scalar dlat); "
                "grid.dlat==0 indicates a tripolar/curvilinear grid, which "
                "uses the ORCA fold for pole handling, not the Fourier filter.")
        lat_v, cos_lat_v = compute_v_face_coords(_grid.lat, dlat)
        pf_grid = SimpleNamespace(
            radius=_grid.radius, n_lon=_grid.n_lon,
            lat=_grid.lat, cos_lat=_grid.cos_lat,
            lat_v=lat_v, cos_lat_v=cos_lat_v)
        kw = dict(
            dt=dt,
            max_wave_speed=cfg.polar_filter.polar_filter_max_wave_speed,
            cutoff_lat_deg=cfg.polar_filter.polar_filter_cutoff_lat_deg,
            safety_factor=cfg.polar_filter.polar_filter_safety_factor,
        )
        mask_c = compute_polar_filter_mask(pf_grid, is_v_face=False, **kw)
        mask_v = compute_polar_filter_mask(pf_grid, is_v_face=True, **kw)

        def _masked_filter_3d(data, wet2d, fmask):
            # wet2d: (n_lat_f, n_lon_f) wet=1/land=0 on the field's stagger.
            wet = wet2d[:, :, None]
            ocean_cnt = jnp.maximum(jnp.sum(wet, axis=1, keepdims=True), 1.0)
            zmean = jnp.sum(data * wet, axis=1, keepdims=True) / ocean_cnt
            filled = jnp.where(wet > 0.0, data, zmean)
            filt = fourier_filter_3d(filled, _grid, fmask)
            # restore exact per-latitude ocean zonal mean (conservation)
            filt_mean = jnp.sum(filt * wet, axis=1, keepdims=True) / ocean_cnt
            filt = filt + (zmean - filt_mean)
            return jnp.where(wet > 0.0, filt, data)

        def _masked_filter_2d(data, wet2d, fmask):
            wet = wet2d
            ocean_cnt = jnp.maximum(jnp.sum(wet, axis=1, keepdims=True), 1.0)
            zmean = jnp.sum(data * wet, axis=1, keepdims=True) / ocean_cnt
            filled = jnp.where(wet > 0.0, data, zmean)
            filt = fourier_filter(filled, _grid, fmask)
            filt_mean = jnp.sum(filt * wet, axis=1, keepdims=True) / ocean_cnt
            filt = filt + (zmean - filt_mean)
            return jnp.where(wet > 0.0, filt, data)

        land = state.land_mask.data    # (n_lat, n_lon) 1=ocean
        u_mask = state.u_mask.data     # (n_lat, n_lon+1)
        v_mask = state.v_mask.data     # (n_lat+1, n_lon)

        T = _masked_filter_3d(state.T.data, land, mask_c)
        S = _masked_filter_3d(state.S.data, land, mask_c)
        # Prescribed-flow lever (STATIC config gate): filter TRACERS only.
        # Filtering the pinned u/v/eta would UNPIN the prescribed circulation
        # (repeated zonal Fourier truncation walks a "frozen" flow away from
        # the held values step after step); the pinned flow needs no polar
        # CFL filter because it never evolves.  T/S keep the stability filter
        # — their advection under "frozen" still crosses converging meridians.
        if getattr(self.config, "prescribed_flow", None) is not None:
            return state._replace(
                T=state.T.replace(data=T), S=state.S.replace(data=S))
        v = _masked_filter_3d(state.v.data, v_mask, mask_v)
        # u-face is periodic with a duplicated wrap column (u[:, n_lon] == u[:, 0]);
        # filter the n_lon-wide interior with the cell mask, then re-append col 0.
        u_int = _masked_filter_3d(state.u.data[:, :-1], u_mask[:, :-1], mask_c)
        u = jnp.concatenate([u_int, u_int[:, 0:1]], axis=1)
        eta = _masked_filter_2d(state.eta.data, land, mask_c)

        return state._replace(
            T=state.T.replace(data=T), S=state.S.replace(data=S),
            u=state.u.replace(data=u), v=state.v.replace(data=v),
            eta=state.eta.replace(data=eta))

    def _apply_freeze_floor(self, state: LatLonCGridOceanState
                            ) -> LatLonCGridOceanState:
        """Floor the SURFACE ocean temperature at the seawater freezing point.

        Sea-ice thermodynamic surrogate (``config.freeze_floor``): an exposed
        surface ocean cell cannot super-cool below the freezing point of
        seawater — the excess heat loss physically goes into ice formation
        (latent heat), which holds SST at freezing.  legoESM carries no
        prognostic ice, so without this cap the high-latitude (esp. Arctic)
        surface over-cools 3-5 C below NEMO (whose LIM sea ice caps SST).  This
        is the same ``jnp.maximum(T, T_freeze)`` clamp the slab oceans apply
        (``simple_ocean.py``).

        SURFACE-ONLY (top cell, k=0): sea ice caps the SST.  The freezing point
        of seawater is depth-dependent (pressure lowers it) and subsurface water
        is in any case above -1.8 C, so a full-column constant floor could mask
        a genuine deep cold anomaly or inject deep heat — clamp only the top
        cell.  Land cells (T=0) are above freezing, so ``maximum`` is a no-op
        there (no wet mask needed).

        CONSERVATION: this is an intentional, bounded NON-conservative heat
        source (the latent heat of the ice that would have formed) — it is NOT
        seen by the conservation fixer (which runs earlier in the step).  Heat
        conservation is therefore deliberately relaxed when ``freeze_floor`` is
        enabled, representing sea-ice formation; off by default so legacy /
        conserving runs are unaffected.
        """
        T = state.T.data
        # Freeze-point floor [degC].  Default ("constant") keeps the historical
        # scalar ``freeze_floor_temp_c`` byte-identical.  A liquidus scheme
        # (``config.freezing.scheme``) instead floors each surface cell at ITS
        # OWN freezing point using the local surface salinity — fresher water
        # freezes warmer, more saline colder (the -1.8 -> ~-1.92 C S-dependence).
        # ``freezing_point`` returns KELVIN and the ocean state T is in degC, so
        # subtract ``constants.T_freeze`` (the 0 degC reference).  ``scheme`` is a
        # static config field => feature-gating branch, not a traced select.
        if self.config.freezing.scheme == "constant":
            floor_c = self.config.freeze_floor_temp_c
        else:
            from legoesm import constants as _consts
            from legoesm.ocean.eos import freezing_point
            S_sfc = state.S.data[..., 0]
            floor_c = (
                freezing_point(S_sfc, 0.0, scheme=self.config.freezing.scheme)
                - _consts.T_freeze
            )
        T_sfc_floored = jnp.maximum(T[..., 0], floor_c)
        T_floored = T.at[..., 0].set(T_sfc_floored)
        return state._replace(T=state.T.replace(data=T_floored))

    def _apply_ew_cyclic_overlap(self, state: LatLonCGridOceanState
                                 ) -> LatLonCGridOceanState:
        """Slave the two longitude HALO columns to their ORCA 2-point
        cyclic-overlap partners (``config.ew_cyclic_overlap``).

        The ORCA tripole grid is periodic east-west with a 2-point overlap:
        column ``0`` duplicates column ``nx-2`` and column ``nx-1`` duplicates
        column ``1`` (same geographic longitude).  The C-grid operators apply
        regular-grid roll-periodicity (period ``nx``), which is OFF BY ONE for
        an ORCA grid (true period ``nx-2``) -- and the eORCA1 mesh marks the
        halo columns LAND, so the east-west seam (lon ~72.5E on eORCA1) carries
        a spurious wall and the two physical seam columns drift apart.

        Re-imposing the overlap at the END of each step makes the seam-adjacent
        physical columns see the correct cross-seam neighbour on the NEXT step:
        the physical seam u-face (``u[:,1]``) gets the true ``(col1 - col_{nx-2})``
        eta-gradient PGF + correct upwind tracers, reconnecting both barotropic
        flow and tracer advection across the seam.

        ALL prognostic fields are slaved (codex adversarial-review HIGH): the
        barotropic solver carries ``U_old``/``u_prime`` forward (velocity is NOT
        recomputed from scratch) and the seam u-face Coriolis term reads the
        ``v`` halo via ``roll(V_bar_c, 1)`` -- so an unslaved velocity halo would
        re-inject the seam every step, defeating the tracer fix.  EXCEPTION:
        under ``config.prescribed_flow`` only T/S are slaved -- this call runs
        AFTER the final re-pin and must not mutate the pinned u/v/eta (see the
        gated return below).

        Column rules (``nx = n_lon`` = cell count):
        * cell-centred (T, S, eta) and v-faces (lat interfaces, also ``nx``
          columns): ``col[0] <- col[nx-2]``, ``col[nx-1] <- col[1]``.
        * u-faces (lon interfaces, ``nx+1`` columns): ``u[:,0] <- u[:,nx-2]``
          (west face of cell 0 == cell nx-2) and ``u[:,nx-1] <- u[:,1]`` (east
          face of physical cell nx-2 == the seam face == ``u[:,1]``).  The last
          face ``u[:,nx]`` is left to the model's internal ``u[:,nx]=u[:,0]``
          wrap -- it feeds only the slaved halo cell, so it is harmless.

        Requires the matching mask/bathy/IC overlap-fill at construction (so the
        derived u/v face masks treat the reconnected seam as ocean and the halo
        columns start consistent).  ORCA-OVERLAP-SPECIFIC: correct only on a grid
        whose first/last columns DUPLICATE columns nx-2 / 1; WRONG on a genuinely
        period-nx regular lat-lon grid.  Gated (default off) -> bit-exact for
        every existing grid/config.
        """
        nx = state.T.data.shape[1]   # n_lon (cell count)

        def _ovl_cell(a):
            # cell-column fields (T, S, eta, v): axis-1 size nx.
            a = a.at[:, 0].set(a[:, nx - 2])
            a = a.at[:, nx - 1].set(a[:, 1])
            return a

        def _ovl_u(a):
            # u-faces: axis-1 size nx+1. Slave the 2 halo seam faces.
            a = a.at[:, 0].set(a[:, nx - 2])
            a = a.at[:, nx - 1].set(a[:, 1])
            return a

        # Prescribed-flow lever (STATIC config gate; codex finding 4): slave
        # TRACERS only — the same treatment as the polar filter.  This runs
        # AFTER the final in-step re-pin, so slaving u/v/eta here could MUTATE
        # the pinned circulation (an entry state whose halo columns don't
        # satisfy the overlap identity would have its "frozen" flow silently
        # rewritten at the seam every step, breaking the exact-hold contract;
        # "zero" is trivially invariant).  The pinned flow needs no seam
        # reconnection anyway — it never evolves, and the construction-time
        # overlap-fill (mask/bathy/IC, documented above) already makes the
        # entry halos consistent.  T/S keep the overlap: their advection under
        # "frozen" still crosses the seam.  Chosen over re-pinning after the
        # call because the pin values are locals of _step_impl — recomputing
        # them at the _step_jitted call site would duplicate the pin logic.
        if getattr(self.config, "prescribed_flow", None) is not None:
            return state._replace(
                T=state.T.replace(data=_ovl_cell(state.T.data)),
                S=state.S.replace(data=_ovl_cell(state.S.data)),
            )

        return state._replace(
            T=state.T.replace(data=_ovl_cell(state.T.data)),
            S=state.S.replace(data=_ovl_cell(state.S.data)),
            eta=state.eta.replace(data=_ovl_cell(state.eta.data)),
            u=state.u.replace(data=_ovl_u(state.u.data)),
            v=state.v.replace(data=_ovl_cell(state.v.data)),
        )

    def _ab2_step(self, state: LatLonCGridOceanState, dt: float,
                  freshwater=None, surface_forcing=None, sponge=None,
                  *, grid=None, vertex_mask=None,
                  t_seconds=None, z_coord=None, config=None, iwm_fields=None) -> LatLonCGridOceanState:
        """Adams-Bashforth-2 outer integrator (Veros's faithful scheme).

        Veros AB2-extrapolates only the EXPLICIT tendency and applies implicit
        vertical mixing ONCE afterward (``core/thermodynamics.py`` tracers,
        ``core/external/solve_stream.py`` momentum)::

            X*       = X^n + (1.5+ε)·ΔX_expl^n − (0.5+ε)·ΔX_expl^{n-1}
            X^{n+1}  = ImplicitVertMix(X*)

        where ``ΔX_expl`` is the explicit-only forward-Euler increment
        (advection, GM/Redi, lateral friction, Coriolis, the split-explicit /
        rigid-lid barotropic solve, freshwater) — NOT including the implicit
        vertical mixing.  This is the FAITHFUL refinement of the prior scheme,
        which AB2-extrapolated the TOTAL increment (implicit mixing included)
        and so was only conditionally stable in the vertical and had to reject
        convective adjustment.  Applying implicit mixing once (it is a
        backward-Euler solve, unconditionally stable) restores unconditional
        vertical stability ⇒ compatible with convective adjustment.

        The prior EXPLICIT increment ``ΔX_expl^{n-1}`` is carried on
        ``{T,S,u,v}_incr_prev``; a missing carry bootstraps ``ΔX_expl^{n-1}=0``
        (a 1.6× first step).  Tracers AB2 the full explicit increment; MOMENTUM
        AB2s only the baroclinic deviation (thickness-weighted depth-mean
        split), KEEPING the barotropic mode from the barotropic solve un-AB2'd
        (the barotropic gravity wave is CFL-stiff and must not be extrapolated).
        Every op is linear in the increments ⇒ differentiable.

        CONSERVATION: like Veros (and like the prior AB2 scheme), this extrapolates
        the tracer CONCENTRATION ``T``, so the area·thickness-weighted heat/salt
        content ``Σ(T·h·area)`` is conserved EXACTLY only when the layer thickness is
        fixed — under ``barotropic_solver="rigid_lid"`` (Veros's streamfunction rigid
        lid; the faithful ACC config) or with ``use_conservation_fixer=True``. Under a
        moving free surface (z*) the concentration extrapolation carries an O(Δη)
        tracer-content drift (small: heat ~1e-7 over 200 closed-channel steps, salt at
        round-off). The forward-Euler path conserves to machine zero regardless.

        K-PROFILE FIDELITY: the implicit-once solve reuses the diffusivity profiles
        returned by ``tendencies`` (from the pre-step state u^n) when the vmix scheme
        surfaces them on ``tend.K_v``/``tend.A_v``; for schemes that don't (e.g. TKE),
        ``_apply_implicit_vertical_mixing`` recomputes them from the state it acts on —
        exactly as the forward-Euler path does (a 2nd-order difference, not a regression).

        ``grid``/``vertex_mask`` (optional, SPMD): default ``None`` →
        ``self.grid``/``self._vertex_mask`` (bit-identical single-device
        path), threaded into every sub-step (``_step_impl``,
        ``_apply_implicit_vertical_mixing``, ``_apply_tke_advection``, the
        conservation fixer); a band-local grid (+ vertex mask) is injected
        by a future ``shard_map`` wrapper.  ``vertex_mask`` is consumed
        only inside ``_step_impl``/``tendencies`` (this method's own body
        reads only the grid), so it is passed straight through.
        """
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        _grid = grid if grid is not None else self.grid
        # Explicit-only forward-Euler step (skips implicit vertical mixing AND the
        # conservation fixer) + the implicit-mixing diffusivity profiles from the
        # tendencies (pre-step state u^n where the vmix scheme surfaces them; else
        # None ⇒ recomputed in _apply_implicit_vertical_mixing, as the FE path).
        state_expl, (K_v_phys, A_v_phys, k33_implicit,
                     surface_tracer_forcing, tke_source,
                     diss_incr, tracer_source) = self._step_impl(
            state, dt, freshwater=freshwater,
            surface_forcing=surface_forcing, sponge=sponge,
            _apply_implicit_vmix=False, grid=_grid, vertex_mask=vertex_mask,
            t_seconds=t_seconds, z_coord=z_coord, config=config, iwm_fields=iwm_fields)
        _tke_prog = self._tke_prognostic_active()
        _tke_old = (state.tke.data if (_tke_prog and state.tke is not None)
                    else None)
        eps = _cfg_b.ab2_epsilon
        a_n, a_p = 1.5 + eps, 0.5 + eps
        mask3 = state.land_mask.data[..., jnp.newaxis]
        u_mask3 = state.u_mask.data[..., jnp.newaxis]
        v_mask3 = state.v_mask.data[..., jnp.newaxis]
        # Partial-cell coords: fold the per-level FACE/CELL activity into the
        # AB2 masks.  Without this, a face that is wet at the surface but
        # CLOSED at depth (variable bathymetry) carries a growing artifact:
        # the rigid lid zeroes the closed-cell velocity each step
        # (u_active_3d) while this reconstruction rebuilds
        # u^{n+1} = u^n + 1.6·Δ^n − 0.6·Δ^{n-1} with Δ^n = −u^n there — the
        # recurrence u^{n+1} = −0.6·u^n + 0.6·u^{n-1}, |r|max = 1.13 ⇒ +13%
        # PER STEP unbounded growth in every closed-face cell (the global_4deg
        # deep-cell blowup; ablation-isolated, scheme-independent).  Masking
        # the AB2 state AND the carried increments pins closed cells at 0 — a
        # stable fixed point.  Pure z-star: masks unchanged ⇒ bit-identical.
        if isinstance(_zc, OceanPartialCellCoordinate):
            _au3, _av3 = compute_face_masks_3d(
                _zc.is_active, _grid)
            u_mask3 = u_mask3 * _au3.astype(u_mask3.dtype)
            v_mask3 = v_mask3 * _av3.astype(v_mask3.dtype)
            mask3 = mask3 * _zc.is_active.astype(mask3.dtype)

        # --- Tracers: AB2 the EXPLICIT increment ---
        # Under ab2_scope="total" (default) ``state_expl.T - state.T`` is the
        # FULL explicit increment (advection + lateral diffusion + GM/Redi);
        # ``diss_incr`` is None ⇒ the algebra below is bit-identical.
        # Under ab2_scope="advective" the dissipative tendencies were WITHHELD
        # from ``state_expl`` (PE + GM/Redi routed into ``diss_incr``), so
        # ``dT_n`` is the ADVECTIVE-only increment that is AB2-extrapolated, and
        # the weight-1.0 dissipative increment ``ΔT_diss`` is ADDED separately:
        #   T^{n+1} = T^n + a_n·ΔT_adv^n − a_p·ΔT_adv^{n-1} + 1.0·ΔT_diss^n
        # (Veros core/thermodynamics.py: AB2 advection, diffusion at weight 1.0).
        dT_n = state_expl.T.data - state.T.data
        dS_n = state_expl.S.data - state.S.data
        dT_p = (state.T_incr_prev.data if state.T_incr_prev is not None
                else jnp.zeros_like(dT_n))
        dS_p = (state.S_incr_prev.data if state.S_incr_prev is not None
                else jnp.zeros_like(dS_n))
        if diss_incr is not None:
            dT_diss_incr, dS_diss_incr, du_diss_incr, dv_diss_incr = diss_incr
        else:
            dT_diss_incr = dS_diss_incr = du_diss_incr = dv_diss_incr = 0.0
        # NB (#517 item 8): NOT routed through ab2_blend.  The original adds
        # the blend INTO ``state.X.data`` as ``((base + a_n·new) - a_p·old)``;
        # ``base + ab2_blend(...)`` re-associates to ``base + (a_n·new −
        # a_p·old)`` and FP addition is non-associative → ~1e-16 byte drift
        # (codex).  Kept inline.  ab2_blend is used only where the blend is a
        # STANDALONE subexpression (rigid-lid ψ, the flux-div / TKE AB2 above).
        T_ab2 = (state.T.data + a_n * dT_n - a_p * dT_p + dT_diss_incr) * mask3
        S_ab2 = (state.S.data + a_n * dS_n - a_p * dS_p + dS_diss_incr) * mask3

        # --- Momentum: AB2 the BAROCLINIC increment; keep the explicit
        #     barotropic mode (CFL-stiff, set by the barotropic solver) ---
        h_k = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, _zc,
            min_water_column_m=_cfg_b.min_water_column_m)
        h_u = min_cell_to_uface(h_k)
        h_v = min_cell_to_vface(h_k, _grid)

        def _split(field, h_face):
            # (#517 item 5: shared depth_mean; floor 1.0e-10 + keepdims
            # passed verbatim → bit-identical.)  Was open-coded as TWO
            # separate sums → fused=False (byte-identity reduction topology).
            bt = depth_mean(field, h_face, 1.0e-10, keepdims=True,
                            fused=False)
            return field - bt, bt          # (baroclinic deviation, barotropic mean)

        # Prescribed-flow lever (STATIC config gate; None default keeps the
        # block below byte-identical).  The momentum AB2 extrapolation is
        # SKIPPED entirely: state_expl.u/v/eta already carry the pin from
        # _step_impl, and consuming u_incr_prev/v_incr_prev here would
        # re-inject discarded-momentum history into the pinned flow (codex
        # finding 2 — stale AB2 carries).  du_n/dv_n = 0 ⇒ the momentum
        # carries stored at the bottom of this method are ZEROS, so the stale
        # carry is never read AND never re-written.  The tracer AB2 (above)
        # is untouched — the T/S physics increments are real under the lever.
        _pflow = getattr(_cfg_b, "prescribed_flow", None)
        if _pflow is not None:
            du_n = jnp.zeros_like(state.u.data)
            dv_n = jnp.zeros_like(state.v.data)
            u_ab2 = state_expl.u.data
            v_ab2 = state_expl.v.data
        else:
            # NB (adversarial-review #4, low severity): the carried du_p was depth-mean-
            # zero under the PREVIOUS step's h_u; applying it at the current h_u injects a
            # spurious barotropic component ≈ a_p·<du_p>_z(current h). Negligible for ACC
            # (O(1 m) eta → O(1e-4 m/s)), and exactly zero under rigid-lid (fixed h);
            # fix for large free-surface motion by re-splitting du_p under the current h_u.
            ubc_n, _ = _split(state.u.data, h_u)
            ubc_e, bt_u_e = _split(state_expl.u.data, h_u)
            vbc_n, _ = _split(state.v.data, h_v)
            vbc_e, bt_v_e = _split(state_expl.v.data, h_v)
            du_n = ubc_e - ubc_n          # baroclinic explicit increment
            dv_n = vbc_e - vbc_n
            du_p = (state.u_incr_prev.data if state.u_incr_prev is not None
                    else jnp.zeros_like(du_n))
            dv_p = (state.v_incr_prev.data if state.v_incr_prev is not None
                    else jnp.zeros_like(dv_n))
            u_ab2 = ((ubc_n + a_n * du_n - a_p * du_p) + bt_u_e) * u_mask3
            v_ab2 = ((vbc_n + a_n * dv_n - a_p * dv_p) + bt_v_e) * v_mask3
            # ab2_scope="advective": add the weight-1.0 DISSIPATIVE momentum
            # increment (lateral friction + bottom drag, evaluated on u^n) as its
            # BAROCLINIC DEVIATION only. The increment's DEPTH-MEAN already forced
            # the barotropic solve (the diss depth-mean is folded into F_slow in
            # _step_impl — Veros solve_stream.py:80's uloc includes du_mix), so the
            # final barotropic mode carries it via ψ; adding the raw increment here
            # would double-count it for one step and then be DISCARDED by the rigid
            # lid (which replaces the incoming depth-mean each step) — the
            # review-caught routing bug that ran the transport away to 3865 Sv.
            # du_diss_incr is 0.0 under "total" ⇒ the _split is a no-op branch.
            if diss_incr is not None:
                du_diss_bc, _ = _split(du_diss_incr, h_u)
                dv_diss_bc, _ = _split(dv_diss_incr, h_v)
                u_ab2 = u_ab2 + du_diss_bc * u_mask3
                v_ab2 = v_ab2 + dv_diss_bc * v_mask3
            u_ab2 = u_ab2.at[:, -1].set(u_ab2[:, 0])   # periodic-lon wrap

        state_ab2 = state_expl._replace(
            T=state_expl.T.replace(data=T_ab2),
            S=state_expl.S.replace(data=S_ab2),
            u=state_expl.u.replace(data=u_ab2),
            v=state_expl.v.replace(data=v_ab2),
        )

        # --- Implicit vertical mixing applied ONCE to the AB2 state (Veros
        #     thermodynamics.py vertmix / solve_stream.py du_mix), with the
        #     diffusivity profiles from the tendencies (see docstring) ---
        # Prognostic TKE under AB2: the TRACER implicit-mixing call is the
        # single authoritative TKE site (tke_old from state.tke, the source
        # assembled by _step_impl from this step's EKE update + K_diss_bot);
        # the updated TKE is stored on the final state below.  The additive-
        # friction MOMENTUM-only call passes tke_old WITHOUT return_tke so its
        # A_v profiles are consistent with the carried TKE (Veros derives
        # kappaM from tke[tau] at step start) but never advances it.  Off ⇒
        # tke args are all None ⇒ bit-identical.
        tke_new_ab2 = None
        if _cfg_b.implicit_vertical_mixing:
            dt_mom = dt / _cfg_b.dt_mom_ratio
            if _cfg_b.momentum_friction_additive:
                # Veros ADDITIVE friction placement (core/friction.py +
                # core/external/solve_stream.py): the implicit vertical-friction
                # increment is evaluated on the PRE-STEP velocity u^n and ADDED
                # to the AB2 state — u^{n+1} = AB2(u) + (BE(u^n) − u^n) — at
                # weight 1.0 (NOT AB2-extrapolated, like the implicit surface
                # forcing).  TRACERS keep the sequential implicit-diffusion-on-
                # the-AB2-state placement in both modes (that IS Veros's tracer
                # placement, core/thermodynamics.py).  The friction solve has
                # zero-flux top/bottom BCs ⇒ the increment's depth-mean is ~0 ⇒
                # the un-AB2'd barotropic mode from the barotropic solver is
                # untouched.  Measured (.physics-validator/momentum_fair/): the
                # additive form collapses the realized momentum-increment L2
                # ratio 4.8→1.6 and lifts corr 0.11→0.44 vs Veros.
                # Post-mixing TKE with the REALIZED K_diss_v: the friction
                # increments live in THIS momentum-only call (BE(u^n) − u^n),
                # so the realized dissipation is computed here and threaded
                # into the tracer call (the authoritative TKE site) —
                # matching Veros, where friction (K_diss_v) runs before
                # integrate_tke (veros.py:266→285). Off ⇒ no extra return ⇒
                # bit-identical.
                _want_kdv = self._tke_realized_kdiss_active()
                # Momentum-only: this call already acts on the step-entry
                # ``state`` (before advection), so its N² is ALREADY Nnow —
                # no n2_tracers override needed (and it must stay that way so
                # friction + tracer N² sources remain consistent under the
                # n2_before_advection flag).
                _fric = self._apply_implicit_vertical_mixing(
                    state, dt, surface_forcing,
                    K_v_phys=K_v_phys, A_v_phys=A_v_phys,
                    dt_mom=dt_mom, do_tracers=False,
                    tke_old=_tke_old,
                    return_K_diss_v=_want_kdv,
                    grid=_grid,
                z_coord=z_coord, config=config, iwm_fields=iwm_fields)
                if _want_kdv:
                    state_fric, _kdiss_v_w = _fric
                else:
                    state_fric, _kdiss_v_w = _fric, None
                du_impl = state_fric.u.data - state.u.data
                dv_impl = state_fric.v.data - state.v.data
                _trac = self._apply_implicit_vertical_mixing(
                    state_ab2, dt, surface_forcing,
                    K_v_phys=K_v_phys, A_v_phys=A_v_phys, K33_iso=k33_implicit,
                    dt_mom=dt_mom,
                    surface_tracer_forcing=surface_tracer_forcing,
                    tracer_source=tracer_source,
                    do_momentum=False,
                    tke_old=_tke_old, tke_source=tke_source,
                    return_tke=_tke_prog,
                    K_diss_v_w=_kdiss_v_w,
                    grid=_grid,
                    # NEMO eosbn2 Nnow N²: step-entry (before-advection) T/S.
                    n2_tracers=self._n2_before_advection_tracers(state, z_coord=z_coord, config=config),
                    # NEMO e3w(Kmm) divisor (#1226 W1): state_ab2 is the
                    # post-AB2 AFTER state; state.eta is NOW (un-rebound
                    # _ab2_step parameter). No-op when the flag is off.
                    eta_now=state.eta.data,
                    u_now=state.u.data, v_now=state.v.data,
                z_coord=z_coord, config=config, iwm_fields=iwm_fields)
                if _tke_prog:
                    state_ab2, tke_new_ab2 = _trac
                else:
                    state_ab2 = _trac
                u_add = (state_ab2.u.data + du_impl) * u_mask3
                v_add = (state_ab2.v.data + dv_impl) * v_mask3
                u_add = u_add.at[:, -1].set(u_add[:, 0])   # periodic-lon wrap
                state_ab2 = state_ab2._replace(
                    u=state_ab2.u.replace(data=u_add),
                    v=state_ab2.v.replace(data=v_add),
                )
            else:
                _seq = self._apply_implicit_vertical_mixing(
                    state_ab2, dt, surface_forcing,
                    K_v_phys=K_v_phys, A_v_phys=A_v_phys, K33_iso=k33_implicit,
                    dt_mom=dt_mom,
                    surface_tracer_forcing=surface_tracer_forcing,
                    tracer_source=tracer_source,
                    tke_old=_tke_old, tke_source=tke_source,
                    return_tke=_tke_prog,
                    grid=_grid,
                    # NEMO eosbn2 Nnow N²: step-entry (before-advection) T/S.
                    n2_tracers=self._n2_before_advection_tracers(state, z_coord=z_coord, config=config),
                    # NEMO e3w(Kmm) divisor (#1226 W1): see the sibling
                    # additive-friction tracer call above.
                    eta_now=state.eta.data,
                    u_now=state.u.data, v_now=state.v.data,
                z_coord=z_coord, config=config, iwm_fields=iwm_fields)
                if _tke_prog:
                    state_ab2, tke_new_ab2 = _seq
                else:
                    state_ab2 = _seq
        if tke_new_ab2 is not None:
            # Veros order: superbee-advection AB2 increment AFTER the implicit
            # solve (tke.py:315-323), advecting the carried state.tke by the
            # pre-step u[tau]; dt is the TRACER dt (Veros dt_tracer — distinct
            # from the solve's dt_mom under async stepping).
            if self._tke_advection_active():
                tke_new_ab2, _dtke_field = self._apply_tke_advection(
                    state, tke_new_ab2, dt, grid=_grid, z_coord=z_coord, config=config)
                state_ab2 = state_ab2._replace(dtke=_dtke_field)
            state_ab2 = state_ab2._replace(
                tke=Field(data=tke_new_ab2, name="tke",
                          dims=("lat", "lon", "level"), units="m^2/s^2"),
            )

        # --- Conservation fixer once, on the final state ---
        if _cfg_b.use_conservation_fixer:
            state_ab2 = ocean_conservation_fixer(
                state_ab2, state, _grid, _zc, _cfg_b,
            )

        # Prescribed-flow lever: FINAL re-pin from state_expl (which carries
        # the pin from _step_impl).  The implicit vertical-mixing call above
        # mixes MOMENTUM too (both the additive-friction and sequential
        # paths), and the volume fixer can shift eta — run-and-discard, same
        # convention as _step_impl's final re-pin.
        if _pflow is not None:
            state_ab2 = state_ab2._replace(
                u=state_ab2.u.replace(data=state_expl.u.data),
                v=state_ab2.v.replace(data=state_expl.v.data),
                eta=state_ab2.eta.replace(data=state_expl.eta.data),
            )

        # Carry the EXPLICIT increments for the next AB2 step (NOT the
        # post-implicit-mixing increment — that is the prior scheme's bug).
        # CARRY SEMANTICS (gated): under ab2_scope="total" (default) dT_n / du_n
        # are the FULL explicit increment (advection + dissipation) ⇒ carry is
        # unchanged bit-identically. Under ab2_scope="advective" the dissipative
        # tendencies were withheld from state_expl, so dT_n / du_n are the
        # ADVECTIVE-only increment ⇒ the carry holds ONLY the advective part
        # (Veros carries ``dtemp`` = advection only; diffusion is never lagged).
        return state_ab2._replace(
            T_incr_prev=Field(data=dT_n * mask3, name="T_incr_prev",
                              dims=state.T.dims, units=state.T.units),
            S_incr_prev=Field(data=dS_n * mask3, name="S_incr_prev",
                              dims=state.S.dims, units=state.S.units),
            u_incr_prev=Field(data=du_n * u_mask3, name="u_incr_prev",
                              dims=state.u.dims, units=state.u.units),
            v_incr_prev=Field(data=dv_n * v_mask3, name="v_incr_prev",
                              dims=state.v.dims, units=state.v.units),
        )

    # HISTORICAL NOTE, kept because the gap it named was real for months.
    # A ``_warn_euler_start_skips_after_reconcile`` method used to live here
    # and fire a RuntimeWarning saying that the forward-Euler start returned
    # before the reconciliation site while "NEMO DOES run mlf_baro_corr on its
    # l_1st_euler step".  That was true, and it is now FIXED rather than
    # announced: there is no early return any more (#1729), so the Euler start
    # reaches this method like every other step.  Nothing warns because
    # nothing is skipped.

    def _apply_after_level_reconcile(self, naa, state, btu_exp, btv_exp,
                                     u_mask3, v_mask3, grid, kaa_eta_raw=None,
                                     z_coord=None, config=None):
        """NEMO ``mlf_baro_corr``, the SECOND depth-mean reconciliation.

        Transcribes ``cfgs/DINO/MY_SRC/stpmlf.F90:578 -> :754-765``.  The
        CALLER must invoke this at NEMO's position: after ``dyn_zdf`` (:396,
        the implicit vertical solve) and before the Asselin filter
        (``dyn_atf_qco``, :613).  Both outer-step paths (``_leapfrog_step``
        and ``_nemo_mlf_step``) call it there, and this is the ONLY
        implementation -- the two paths cannot drift apart on it.

        Selected by ``BarotropicConfig.barotropic_after_reconcile``.  ``"off"``
        (the default) returns ``naa`` unchanged, with no array work, so the
        default path is bit-identical; an unknown value raises here, on the
        static config value.

        WHAT IT ACTUALLY DOES, in order of how much it moves -- measured, not
        assumed, because the option's name misleads on this:

        1. It DISCARDS whatever column mean the implicit vertical solve
           deposited, which is ~99.8% of its effect.  This is the fidelity row:
           NEMO throws that deposit away every step (measured at +17.9 m3/s2
           per southern u-row on the 90-day DINO twin, 12x the realized
           spin-up rate), and legoESM otherwise keeps a residue of it.
        2. It re-pins the column mean by executing the LIVE Kaa face-thickness
           reduction and independently materialized reciprocal.  Their QCO
           factor cancels algebraically but not at the final ULP; round 49
           measured the executed form as bit-exact and the cancelled form as
           pointwise debt.

        THE TARGET is ``btu_exp``/``btv_exp``, the depth mean the barotropic
        solve produced -- i.e. whichever substep average
        ``barotropic_reconcile_target`` selected.  NEMO commits ``uu_b(Kaa)``,
        its PRIMARY velocity-weighted boxcar (``dynspg_ts.F90:978-980,:1001``
        under ``ln_dynadv_vec``), which is legoESM's ``"velocity_avg"``.  Under
        ``"transport_avg"`` this installs legoESM's analogue of NEMO's
        ``un_adv/hu`` instead -- and that is the average NEMO explicitly
        DELETES again at ``stpmlf.F90:788``, so ``transport_avg`` +
        ``nemo_mlf_baro_corr`` is NOT the faithful combination.  Matching NEMO
        on this row needs BOTH ``velocity_avg`` and ``nemo_mlf_baro_corr``.

        Round 49 retracted the earlier claim that the QCO factor can be
        cancelled in production.  It cancels algebraically, but NEMO executes
        the live thickness reduction and the independently built reciprocal
        before that cancellation.  The final few ULPs depend on that order.
        ``kaa_eta_raw`` therefore carries the pre-projection Kaa SSH as a
        transient auxiliary from ``_step_impl``; it never enters the model
        state or restart schema.
        """
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        scheme = validate_after_reconcile(
            _cfg_b.barotropic.barotropic_after_reconcile)
        if scheme == "off":
            return naa
        if kaa_eta_raw is None:
            raise ValueError(
                "nemo_mlf_baro_corr requires the raw pre-projection Kaa SSH "
                "from the split-explicit solve")
        # REFERENCE ladder (NEMO e3u_0/e3v_0): the free-surface scaling cancels
        # inside NEMO's own reconciliation, so the faithful weight carries no
        # eta at all. Built from eta = 0 so partial-cell thicknesses survive
        # while the z-star Jacobian does not.
        h_k_ref = compute_layer_thickness(
            jnp.zeros_like(naa.eta.data), state.H_bathy.data, _zc,
            min_water_column_m=_cfg_b.min_water_column_m)
        # CELL -> FACE BY THE ARITHMETIC MEAN, via the SHARED
        # ``interp_cell_to_uface``/``interp_cell_to_vface`` (grids.
        # operators_latlon_cgrid) -- NOT a local reimplementation: those pad
        # the lon halo with ``pad_lon_cgrid`` and the lat halo with
        # ``pad_ns_scalar``, so they span a 2-D partition cut and the north
        # fold, which a rank-local ``jnp.roll`` would not.
        #
        # WHY THE MEAN, AND EXACTLY HOW FAR THAT GOES.  This kernel weights by
        # NEMO's REFERENCE face thickness ``e3u_0``, and NEMO BUILDS THAT TWO
        # DIFFERENT WAYS.  Which one is right is a property of the DOMAIN
        # BUILDER that produced the grid, NOT of ``ln_zco`` vs ``ln_zps``:
        #   * DINO / usrdef grids average --
        #     ``pe3u = 0.50_wp * ( pe3t(ji,jj,jk) + pe3t(ji+1,jj,jk) )``,
        #     ``cfgs/DINO/MY_SRC/zgr_lib.F90:231``.  ``usr_def_zgr`` reaches
        #     it via ``zgr_sco_mi96`` on BOTH its ``ld_zco`` branch and its
        #     else branch (``usrdef_zgr.F90:108-139``), so a DINO-built grid
        #     averages whatever the ln_zco/ln_zps flag says.
        #   * DOMAINcfg partial-step grids take the MINIMUM --
        #     ``e3u_0 = MIN( e3t_0(ji,jj,jk), e3t_0(ji+1,jj,jk) )``,
        #     ``tools/DOMAINcfg/src/domzgr.F90:1166``, inside ``SUBROUTINE
        #     zgr_zps`` (:987).  So ``min_cell_to_uface`` is NOT merely the
        #     MOM6/MITgcm ``hFacW`` convention -- it is ALSO NEMO's own rule
        #     for that grid class.  (An earlier revision of this comment said
        #     min was "a DIFFERENT quantity" full stop.  RETRACTED.)
        # The mean is therefore the faithful choice for the ONLY card that
        # ships this option (``nemo_dino_kamm_mlf``, a DINO-built grid).
        #
        # WHAT THIS CHANGE ACTUALLY DOES, stated without overclaiming: it is
        # INERT on every card that ships the option, EXACTLY -- ``masked_zco``
        # makes each cell thickness ``h_partial in {0, dz_ref[k]}``, so at any
        # face the both-cells-wet mask leaves open, both cells carry
        # ``dz_ref[k]`` and min == mean bit-for-bit (measured over the whole
        # DINO geometry; closed faces differ but the kernel zeroes them).  It
        # would MATTER, and would be WRONG, only on a DOMAINcfg partial-step
        # grid -- which no card enabling this option uses.  Selecting the rule
        # from the builder instead of hardcoding it is tracked as debt D2.5
        # (``docs/ocean/fidelity/dino_outstanding_fidelity_debt.md``); it is
        # deliberately NOT built here, because no such card exists and the two
        # rules are indistinguishable on every card that does, so the selector
        # would ship untested.
        #
        # RELATED, NOT FIXED HERE (same debt row): the TARGET this installs,
        # ``btu_exp``, comes from the barotropic solve, whose column depth is
        # built with the MIN rule.  On a grid where the two rules differ, the
        # weighting and the target would disagree end-to-end.  On every card
        # that ships this option they are the same number.
        #
        # Work on NEMO's native east/north faces, then map once to legoESM's
        # redundant west/south boundary layout.  A NEMO bridge supplies the
        # raw mesh operands.  Generated grids use the same arithmetic from
        # their reference face ladder and native C-grid metrics; the active
        # DINO certification takes the raw branch.
        dtype = naa.u.data.dtype
        nlev = naa.u.data.shape[-1]
        raw_names = (
            "nemo_e3t_0", "nemo_hu_0", "nemo_hv_0", "nemo_e1e2t",
            "nemo_e1e2u", "nemo_e1e2v",
        )
        raw_refs = tuple(getattr(_zc, name, None) for name in raw_names)
        raw_umask = jnp.asarray(u_mask3[:, 1:, :], dtype=dtype)
        raw_vmask = jnp.asarray(v_mask3[1:, :, :], dtype=dtype)
        if all(value is not None for value in raw_refs):
            e3t0, hu0, hv0, area_t, area_u, area_v = (
                jnp.asarray(value, dtype=dtype) for value in raw_refs)
            e3u0 = e3t0[..., :nlev]
            e3v0 = e3t0[..., :nlev]
        else:
            e3u0 = interp_cell_to_uface(h_k_ref)[:, 1:, :]
            e3v0 = interp_cell_to_vface(h_k_ref, grid)[1:, :, :]
            b = jax.lax.optimization_barrier
            hu0 = jnp.zeros_like(kaa_eta_raw, dtype=dtype)
            hv0 = jnp.zeros_like(kaa_eta_raw, dtype=dtype)
            for jk in range(nlev):
                hu0 = b(hu0 + b(e3u0[..., jk] * raw_umask[..., jk]))
                hv0 = b(hv0 + b(e3v0[..., jk] * raw_vmask[..., jk]))
            geom_grid = ensure_geometry(grid)
            area_t = geom_grid.area_T
            area_u_raw = (geom_grid.dx_u * geom_grid.dy_u)[:, 1:]
            area_v_raw = (geom_grid.dx_v * geom_grid.dy_v)[1:, :]
            area_u = jnp.where(raw_umask[..., 0] > 0.0, area_u_raw, 1.0)
            area_v = jnp.where(raw_vmask[..., 0] > 0.0, area_v_raw, 1.0)

        live = nemo_qco_live_face_geometry_from_operands(
            kaa_eta_raw, e3u0, e3v0, raw_umask, raw_vmask,
            hu0, hv0, area_t, area_u, area_v)
        u_native = nemo_literal_after_level_reconcile(
            naa.u.data[:, 1:, :], live.e3u, live.r1_hu,
            btu_exp[:, 1:, :], raw_umask)
        v_native = nemo_literal_after_level_reconcile(
            naa.v.data[1:, :, :], live.e3v, live.r1_hv,
            btv_exp[1:, :, :], raw_vmask)
        u_rec = jnp.concatenate([u_native[:, -1:, :], u_native], axis=1)
        v_rec = jnp.concatenate(
            [jnp.zeros_like(v_native[:1]), v_native], axis=0)
        return naa._replace(u=naa.u.replace(data=u_rec),
                            v=naa.v.replace(data=v_rec))

    def _leapfrog_step(self, state: LatLonCGridOceanState, dt: float,
                       freshwater=None, surface_forcing=None, sponge=None,
                       *, grid=None, vertex_mask=None,
                       t_seconds=None,
                       external_tracer_rate=None, z_coord=None, config=None, iwm_fields=None) -> LatLonCGridOceanState:
        """NEMO Modified-Leap-Frog step (``stp_MLF``, ``stpmlf.F90``, key_qco DINO).

        ``external_tracer_rate`` (#1492): see ``step()`` docstring. Threaded
        ONLY into the advective Nnn pass's ``_step_impl`` call below (as
        ``_external_tracer_rate``) — matching NEMO's ``tra_sbc(kstp, Nnn, ts,
        Nrhs)`` call (stpmlf.F90:342), which runs once per step at Nnn, NOT
        on the separate Nbb dissipative pass this method also runs.

        Three time levels — before ``Nbb`` (t-dt, carried on the state's
        ``{u,v,T,S,eta}_before`` fields), now ``Nnn`` (t, = ``state``), after
        ``Naa`` (t+dt).  One step is (transcribed file:line from NEMO 5.0.2):

          * Explicit RHS evaluated at ``Nnn`` (advection ``dyn_adv``, vorticity+
            Coriolis ``dyn_vor`` EEN ``(f+zeta)/e3f`` — here via
            ``vorticity_scheme="een_total"`` so Coriolis is IN the RHS, NOT a
            Matsuno rotation — lateral mixing ``dyn_ldf``, HPG ``dyn_hpg``,
            surface-pressure-gradient ``dyn_spg``; stpmlf.F90:248-256).
          * Explicit leap-frog combine over ``rDt = 2dt``:
            ``X(Naa) = X(Nbb) + 2dt·RHS(Nnn)`` (NEMO ``rDt=2·rn_Dt``, stpmlf.F90:468).
          * Implicit vertical friction+diffusion (``dyn_zdf``/``tra_zdf``) as a
            backward-Euler solve over ``rDt=2dt`` on the after-state
            (stpmlf.F90:267,370).
          * Robert-Asselin time filter on the NOW fields (the PLAIN RA filter, NOT
            Robert-Asselin-Williams — DINO uses ``rn_atfp`` only; dynatf_qco.F90:144,
            traatf_qco.F90:209, sshwzv ``ssh_atf``):
            ``X(Nnn)_f = X(Nnn) + gamma·(X(Nbb) - 2·X(Nnn) + X(Naa))`` with
            ``gamma = config.asselin_gamma = rn_atfp = 0.1``.
          * Swap ``Nbb <- X(Nnn)_f``, ``Nnn <- X(Naa)`` (stpmlf.F90:403-406) —
            realised by returning ``Naa`` with the filtered now-fields stored on
            the ``_before`` fields.
          * FIRST step (``Nbb is None``): a forward-Euler start over ``dt`` with NO
            filter (NEMO ``l_1st_euler``, ``rDt=rn_Dt``; the filter is guarded by
            ``IF(.NOT. l_1st_euler)``); the pre-step now-state becomes ``Nbb`` for
            step 2.

        REUSE / STRUCTURE.  This models ``_ab2_step``: it drives ``_step_impl`` at
        ``_apply_implicit_vmix=False`` to obtain the explicit-only increment over
        ``rDt=2dt`` (evaluated at ``Nnn``) plus the diffusivity profiles, forms the
        leap-frog combine ``Naa = Nbb + (X_expl - Nnn)`` at the top level (an affine
        base-shift from the forward-Euler base ``Nnn`` to the leap-frog base
        ``Nbb``; ``eta`` then leap-frogs, ``eta(Naa)=eta(Nbb)-2dt·div(Hu_avg)``),
        applies implicit vertical mixing ONCE over ``rDt=2dt``, then the RA filter.

        FAITHFULNESS RESIDUALS (documented, O(dη) per step; NOT hidden):
          1. **CLOSED (residual #1, THE dt=2700 blocker).** The split-explicit
             barotropic INTEGRATION is now seeded from the BEFORE level (Nbb ssh
             + Nbb depth-mean transport) via
             ``_barotropic_before_state=(eta_before, u_before, v_before)`` on the
             advective ``_step_impl`` call, so the fast free-surface mode
             leap-frogs n-1 → n+1 exactly as NEMO ``dyn_spg_ts`` under
             ``ln_bt_fw=.FALSE.`` (dynspg_ts.F90:494-503:
             ``sshn_e=pssh(Kbb)/un_e=puu_b(Kbb)/vn_e=pvv_b(Kbb)``); the frozen
             slow forcing stays at NOW (NEMO ``zu_frc``), and the 3-D depth-mean
             replacement keeps the NOW velocity.  Ties the barotropic mode into
             the SAME leap-frog as the baroclinic deviation so the outer RA
             filter (step 5) damps the barotropic 2Δt computational mode.  This
             ALONE moved the dt=2700 blow-up from ~step 26 to ~step 65; the
             remaining 2Δx equatorial checkerboard is damped by NEMO's
             ``nn_bt_flt=2`` ``ts_bck_interp`` temporal dissipation
             (``barotropic_time_filter="nemo_boxcar_ab3"``, set on
             ``nemo_dino_kamm_mlf``): the AB3 velocity predictor + the α=0 ssh
             half-step-back interpolation (0.614/0.285/0.088/0.013) — the
             built-in AM4 dissipation the FE numerical damping was previously
             supplying.  With both, the full 180-day dt=2700 run on the bridged
             NEMO mesh is STABLE (eta bounded, no NaN, gridscale eta fraction
             pinned ~0.05).
          2. **CLOSED.** The tracer RA filter is now the z* THICKNESS-WEIGHTED
             content form (NEMO ``tra_atf_qco_lf``, traatf_qco.F90:295-341):
             filter ``(e3t·T)``/``(e3t·S)`` using the layer thickness at the
             before/now/after ssh, then divide by the thickness from the
             Asselin-filtered ssh (``_thickness_weighted_asselin``).  Conserves
             globally-integrated heat/salt content under the moving z*
             coordinate (the old concentration form drifted by O(dη)).  Momentum
             (ln_dynadv_vec) + ssh stay the plain RA form, matching NEMO.
          3. **CLOSED (this iteration).** The barotropic Coriolis is now the LIVE
             per-substep enstrophy-conserving EEN form (NEMO ``dyn_cor_2D``):
             ``barotropic_coriolis_split="live"`` + ``barotropic_coriolis="een"``
             (set on ``nemo_dino_kamm_mlf``) removes the pre-step 2D barotropic
             Coriolis from ``F_slow`` (dynspg_ts.F90:296-300) and re-applies the
             SAME EEN stencil live each substep on the evolving transport (:689).
             The een-total guard is relaxed for this specific EEN-stencil pairing
             (the subtraction cancels the live term). NOTE: this did NOT unblock
             dt=2700 — see the BLOCKER note; the null mode was not the dt=2700
             driver.
          4. **CLOSED (this iteration).** NEMO evaluates the EXPLICIT LATERAL
             DIFFUSION on the BEFORE level Nbb — ``dyn_ldf(Kbb)`` (dynldf.F90:69
             ``dyn_ldf_lap(...,puu(:,:,:,Kbb),...)``) and ``tra_ldf(Kbb)``
             (traldf.F90:97-98 / traldf_iso ``pts(...,Kbb)``), i.e.
             FORWARD-in-time — precisely because a CENTERED (leap-frog) diffusion
             term is UNCONDITIONALLY UNSTABLE. This step now drives ``_step_impl``
             in "advective" scope (``_ab2_scope_override="advective"``) on BOTH the
             Nnn pass (advection/Coriolis-een_total/HPG, diss WITHHELD) and a
             SECOND Nbb pass whose dissipative increment (lateral viscosity +
             iso-neutral Redi + GM/eiv) alone is kept and applied forward at 2dt:
             ``Naa = Nbb + 2dt·adv(Nnn) + 2dt·diss(Nbb)``.  The GM/Redi
             destabiliser is thus at Nbb; the leap-frog-centred diffusion
             instability is removed.  MOMENTUM diss is added as its baroclinic
             deviation only (F_slow under advective scope carries no diss
             depth-mean; the barotropic lateral friction is handled by the mode's
             solve + the A2 biharmonic — the ``_ab2_step`` routing).
          5. **BAROTROPIC SUBSTEP CFL (fixed this iteration).** The barotropic
             mode integrates over rDt=2dt; the config ``n_barotropic_substeps`` is
             calibrated for the dt-window forward-Euler path, so without rescaling
             the substep length DOUBLES (180 s vs 90 s at dt=2700) and the
             split-explicit free surface blows up in ~4 steps. The two ``_step_impl``
             passes here pass ``_barotropic_substep_scale=2`` so the substep length
             (and barotropic CFL) match the FE path exactly.
        DT=2700 BLOCKER — RESOLVED (residual #1 + #1b). The dt=2700 leap-frog now
        runs the FULL 180 days on the bridged NEMO DINO mesh with NO NaN, physical
        T, eta bounded (~0.6 m), gridscale eta fraction pinned ~0.05 — via TWO
        faithful pieces, no non-NEMO stabiliser:
          (#1) the split-explicit barotropic INTEGRATION seeded from the BEFORE
               level (Nbb) — see residual #1 above (dynspg_ts.F90:494-503). Alone
               this moved the blow-up from ~step 26 to ~step 65 (the equatorial
               2Δx eta checkerboard, previously suppressed by FORWARD-Euler
               numerical damping, then grows under the neutral leap-frog).
          (#1b) NEMO's nn_bt_flt=2 ts_bck_interp TEMPORAL DISSIPATION
               (``barotropic_time_filter="nemo_boxcar_ab3"``): the AB3 velocity
               predictor + the α=0 ssh half-step-back interpolation
               (0.614/0.285/0.088/0.013, dynspg_ts.F90:1698-1701) — the built-in
               AM4 dissipation that damps the 2Δx barotropic gravity-wave mode the
               leap-frog does not. With it the gridscale eta fraction stays ~0.05
               (was climbing to >0.2 then exploding).
        The 180-day dt=2700 comparison vs NEMO RUN_TRAJ is UNBLOCKED and DONE
        (controlled, sole variable FE→leap-frog): BSF range ratio 2.65× (leap-frog)
        vs 2.62× (FE) vs 1.0 (NEMO), SST/T300/SSH corr ≥0.99 both — the leap-frog
        time integrator (node 19, the last structural difference) is NOT the driver
        of the residual BSF over-strength (the characterised equatorial f→0
        core-dynamics amplification). Residual #2 (tracer RA filter) is now
        CLOSED (thickness-weighted content form, see residual #2 above).
        """
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        from legoesm.ocean.state import Field
        _grid = grid if grid is not None else self.grid

        # --- FIRST step: NEMO's l_1st_euler start.
        #
        # THIS IS THE SAME PROGRAM, NOT A SECOND PATH.  ``stp_MLF`` has no
        # first-step variant: on ``l_1st_euler`` it sets ``rDt = rn_Dt``
        # (stpmlf.f90:131-133) and runs statement for statement as on any
        # other step, and ``istate.F90:97-99/135-137`` has already copied the
        # initial T/S/u/v onto BOTH time-level slots so ``Kbb == Kmm``, which
        # makes the leap-frog combine degenerate to forward Euler on its own.
        # Only three statements behave differently, and each is guarded INSIDE
        # its own routine, not at the call: the ssh, tracer and momentum
        # Asselin filters (sshwzv.f90:443, traatf_qco.f90:152,
        # dynatf_qco.f90:164).
        #
        # WHAT THIS REPLACED, and why (#1729).  Until now this branch called
        # ``_step_impl`` once and RETURNED, skipping everything the body below
        # does.  Three consequences, ordered by MEASURED size against NEMO's
        # own one-step-from-rest record (scripts/validate/ocean_fidelity/
        # dino_1226/step1_euler_term_attribution.py):
        #
        #  1. THE WHOLE SURFACE TRACER TENDENCY WAS DROPPED, and this is the
        #     one that mattered.  The early return did not forward
        #     ``external_tracer_rate``, which is how the card delivers surface
        #     heat and salt under ``surface_tendency_placement="leapfrog_rhs"``
        #     (NEMO's ``tra_sbc`` at Nnn, stpmlf.f90:460).  The first step of
        #     every from-rest run therefore had NO surface forcing at all.
        #     Withholding the rate again reproduces the old residual exactly:
        #     T 1.55e-3 K rms, vs 5.00e-6 K with it -- 310x, and 12% of the
        #     size of NEMO's own first step.
        #  2. ``mlf_baro_corr`` never ran (stpmlf.f90:534, guarded on
        #     ``ln_dynspg_ts`` ALONE, so NEMO runs it on l_1st_euler too), nor
        #     did the Kmm velocity cycle (dynspg_ts.f90:1003 installs,
        #     stpmlf.f90:720 removes).  This is what the deleted RuntimeWarning
        #     named.  Measured, it moves ONLY velocity -- T/S/eta by exactly
        #     0.0, because it is called after ``tra_zdf`` (:507) and writes
        #     only puu/pvv -- and it moves u AWAY from NEMO by ~3% of the
        #     residual.  Faithful and slightly worse is a signal, not a reason
        #     to revert.
        #  3. The rest of the body: the leap-frog combine, the Nbb dissipative
        #     pass, the before-level barotropic seed, the Kbb-based FCT
        #     bounds.  On this step Nbb == Nnn so these are algebraically
        #     degenerate, but not bitwise; they own the small eta/u movement
        #     that neither 1 nor 2 accounts for.
        #
        # Making the start a PARAMETERISATION of the body -- rDt, the
        # barotropic window scale, and whether the filters run -- closes all
        # three at once and, more importantly, removes the second path that
        # could drift from the first.  There is no knob: the old behaviour is
        # gone.
        #
        # The before==now seed is written onto ``state`` here (c0dd53736's
        # seed, now non-local) because the whole body reads the before level:
        # the leap-frog combine, the Nbb dissipative pass, the barotropic
        # before-seed, the FCT bounds, and the rn2b/Burchard-shear consumers
        # inside ``_step_impl``.  A bridged/restart state never reaches this
        # branch -- its ``u_before`` is already populated -- so the seed only
        # ever applies to a genuine from-rest state, which is exactly NEMO's
        # own cold start.
        _euler_start = state.u_before is None
        if _euler_start:
            state = state._replace(
                u_before=state.u, v_before=state.v, T_before=state.T,
                S_before=state.S, eta_before=state.eta,
            )

        # --- LEAP-FROG + Asselin.
        # rDt = rn_Dt on the Euler start, 2*rn_Dt otherwise (stpmlf.f90:131).
        rdt = (1.0 if _euler_start else 2.0) * dt
        gamma = _cfg_b.asselin_gamma
        _tke_n2_bundle = self._tke_step_entry_n2_bundle(
            state, z_coord=_zc, config=_cfg_b)
        u_mask3 = state.u_mask.data[..., jnp.newaxis]
        v_mask3 = state.v_mask.data[..., jnp.newaxis]
        cmask = state.land_mask.data
        mask3 = cmask[..., jnp.newaxis]
        # Partial-cell coords: fold the per-level FACE/CELL activity into the
        # masks (identical to _ab2_step).  _step_impl reconstructs u at EVERY
        # level as u'·u_mask_2d + U_bar, so a face wet at the surface but CLOSED
        # at depth carries the barotropic mean U_bar (not the old value) in
        # ``state_expl``; the leap-frog combine below would then inject a
        # spurious closed-cell tendency (the recurrence u^{n+1}=u^{n-1}-u^n,
        # |r|=1.618/step — the same deep-cell blow-up the AB2 3-D mask cures).
        # Masking the combine AND the Asselin filter pins closed cells at 0.
        # Pure z-star: masks unchanged ⇒ bit-identical.
        if isinstance(_zc, OceanPartialCellCoordinate):
            _au3, _av3 = compute_face_masks_3d(_zc.is_active, _grid)
            u_mask3 = u_mask3 * _au3.astype(u_mask3.dtype)
            v_mask3 = v_mask3 * _av3.astype(v_mask3.dtype)
            mask3 = mask3 * _zc.is_active.astype(mask3.dtype)

        # 1. ADVECTIVE Nnn pass over rDt=2dt (skip implicit vmix + fixer).  Run in
        #    "advective" scope so the DISSIPATIVE tendencies (lateral viscosity,
        #    iso-neutral Redi, GM/eiv) are WITHHELD from ``state_expl`` and land on
        #    the returned diss_incr — NEMO evaluates the explicit lateral diffusion
        #    forward-in-time on the BEFORE level (dyn_ldf/tra_ldf(Kbb),
        #    dynldf.F90:69 / traldf.F90:97-98), NOT leap-frog-centred (a centred
        #    diffusion is unconditionally unstable — residual #4).  This pass's own
        #    diss_incr (the Nnn evaluation) is DISCARDED; we keep only the advective
        #    increment, the barotropic/eta solve, and the Nnn diffusivity profiles
        #    (K_v/A_v/K33 for the single implicit vertical solve — NEMO avm/avt are
        #    the "now" vertical mixing, correctly at Nnn).
        # The barotropic mode integrates over rDt, so its substep count is
        # scaled ×(rDt/dt) to hold the substep length (barotropic CFL) at the
        # FE-path value — see ``_barotropic_substep_scale`` in ``_step_impl``.
        # rDt scale: 1 on the Euler start (rDt=dt), 2 on a leap-frog step.
        # On the Euler start this ALSO reproduces NEMO's own averaging window:
        # ``ll_fw_start`` is forced .TRUE. on the l_1st_euler step
        # (dynspg_ts.f90:230-232), moving the boxcar centre from 2*nn_e to
        # nn_e. VERIFIED against NEMO's ts_wgt at DINO's resolved nn_e=23:
        # weights agree to 0.0, transport weights to 1.4e-17, both windows.
        _baro_scale = 1 if _euler_start else 2
        # Seed the split-explicit barotropic INTEGRATION from the BEFORE level
        # (Nbb ssh + transport) so the fast free-surface mode leap-frogs
        # n-1 → n+1 (NEMO ln_bt_fw=.FALSE. centred barotropic, dynspg_ts.F90:
        # 494-503 sshn_e=pssh(Kbb)/un_e=puu_b(Kbb)/vn_e=pvv_b(Kbb)); the frozen
        # slow forcing stays at NOW (NEMO zu_frc). This ties the barotropic mode
        # into the SAME leap-frog as the baroclinic deviation, so the outer
        # Robert-Asselin filter (step 5) damps the barotropic 2Δt computational
        # mode — the FE base-shift-from-Nnn.eta was the dt=2700 CFL blocker
        # (residual #1). Only the advective (Nnn) pass drives the live barotropic
        # solve; the Nbb diss pass discards its barotropic result.
        state_expl, (K_v_phys, A_v_phys, k33_implicit, surface_tracer_forcing,
                     tke_source, _diss_incr_nn, tracer_source,
                     kaa_eta_raw, kaa_hu_avg, kaa_hv_avg) = self._step_impl(
            state, rdt, freshwater=freshwater, surface_forcing=surface_forcing,
            sponge=sponge, _apply_implicit_vmix=False, grid=_grid,
            vertex_mask=vertex_mask, t_seconds=t_seconds,
            _ab2_scope_override="advective",
            _barotropic_substep_scale=_baro_scale,
            _barotropic_before_state=(
                state.eta_before.data, state.u_before.data,
                state.v_before.data),
            # FCT/Zalesak monotonicity base = BEFORE level (Kbb): the limited
            # advective increment is applied to Nbb below (T_naa = T_before +
            # (state_expl.T − state.T)), so the FCT bounds must be Kbb-based
            # (NEMO nonosc(Kbb), p2dt=2dt) — else the FE-certified (Nnn,dt)
            # bounds admit cold undershoot at high-lat wall fronts under 2dt.
            # KNOWN GAP (2026-08-10 h_new certification fix): the thickness
            # FCT certifies against is the Nnn-eta h_k threaded through
            # ``_step_impl``, and the applied update is this pass's increment
            # RE-BASED onto Nbb by the combine below — no single-step
            # certificate covers that composition, so leapfrog FCT bounds
            # are APPROXIMATE (O(dη) slack; NEMO-exact needs e3t(Kbb)
            # content + e3t(Kaa) division threaded into the tracer step).
            # The euler path is exactly certified.
            _fct_tracer_before=(
                state.T_before.data, state.S_before.data),
            # #1492 DINO surface_tendency_placement="leapfrog_rhs": fold the
            # externally-supplied surface tracer RHS into THIS (Nnn advective)
            # pass only, matching tra_sbc's Nnn-only call — see this method's
            # docstring and the ``step()`` param doc.
            _external_tracer_rate=external_tracer_rate,
            _tke_n2_bundle_override=_tke_n2_bundle,
            _return_raw_kaa_qco=True,
            z_coord=z_coord, config=config, iwm_fields=iwm_fields)
        # 1b. DISSIPATIVE Nbb pass — evaluate dyn_ldf(Kbb)/tra_ldf(Kbb) + the GM/eiv
        #     trend on the BEFORE state and keep ONLY its dissipative increment
        #     (2dt·diss(Nbb), applied forward-in-time). The GM/Redi destabiliser is
        #     therefore evaluated at Nbb (residual #4). Reusing the full
        #     ``_step_impl`` (rather than a bespoke diff-only path) keeps the
        #     lateral-viscosity + iso-neutral-Redi + GM/eiv numerics in ONE place
        #     (no duplicated dissipation); the pass's advective state + barotropic
        #     solve are discarded. The lateral-friction / lateral-tracer diff is
        #     evaluated on Nbb's raw u/T; GM/Redi on the Nbb-level advected tracer
        #     (T_mid≈Nbb) + the solved eta. None reads the AB2 momentum/tracer
        #     carries (u_incr_prev/…, consumed only in _ab2_step), so the stale
        #     Nnn carries cannot leak into diss_incr_bb.
        nbb = state._replace(
            u=state.u_before, v=state.v_before, T=state.T_before,
            S=state.S_before, eta=state.eta_before)
        _, (_kbb0, _kbb1, _kbb2, _kbb3, _kbb4, diss_incr_bb,
            _kbb6) = self._step_impl(
            nbb, rdt, freshwater=freshwater, surface_forcing=surface_forcing,
            sponge=sponge, _apply_implicit_vmix=False, grid=_grid,
            vertex_mask=vertex_mask, t_seconds=t_seconds,
            _ab2_scope_override="advective",
            _barotropic_substep_scale=_baro_scale,
            _tke_n2_bundle_override=_tke_n2_bundle,
            z_coord=z_coord, config=config, iwm_fields=iwm_fields)

        # 2. Explicit combine.  MOMENTUM: leap-frog the BAROCLINIC deviation only
        #    (u'(Naa) = u'(Nbb) + 2dt·RHS'), and take the BAROTROPIC mode + eta
        #    DIRECTLY from the forward split-explicit barotropic solve
        #    (``state_expl``) — the CFL-stiff external gravity wave must NOT be
        #    leap-frog-extrapolated (that is a |G|~2/step free-surface blow-up).
        #    This mirrors NEMO ``dyn_spg_ts`` + ``mlf_baro_corr`` (the barotropic
        #    mode comes from the split-explicit solve, then replaces the 3D
        #    depth-mean, stpmlf.F90:392/514-524) and legoESM ``_ab2_step`` (which
        #    keeps the barotropic mode un-AB2'd).  TRACERS leap-frog fully
        #    (T(Naa)=T(Nbb)+2dt·(flux-form increment); NEMO ``tra_atf``).
        h_k = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, _zc,
            min_water_column_m=_cfg_b.min_water_column_m)
        h_u = min_cell_to_uface(h_k)
        h_v = min_cell_to_vface(h_k, _grid)

        def _split(field, h_face):
            bt = depth_mean(field, h_face, 1.0e-10, keepdims=True, fused=False)
            return field - bt, bt      # (baroclinic deviation, barotropic mean)

        ubc_now, _ = _split(state.u.data, h_u)
        ubc_bef, _ = _split(state.u_before.data, h_u)
        ubc_exp, btu_exp = _split(state_expl.u.data, h_u)
        vbc_now, _ = _split(state.v.data, h_v)
        vbc_bef, _ = _split(state.v_before.data, h_v)
        vbc_exp, btv_exp = _split(state_expl.v.data, h_v)
        # Nbb dissipative increment (2dt·diss(Nbb)) — advective-scope diss_incr:
        #   (dT_diss, dS_diss, du_diss, dv_diss).  MOMENTUM: BAROCLINIC deviation
        #   ONLY — the diss depth-mean is intentionally NOT injected into the
        #   split-explicit barotropic mode (F_slow under advective scope carries no
        #   diss; the mode's own solve + the A2 biharmonic handle the barotropic
        #   lateral friction; adding the raw depth-mean double-counts and is
        #   discarded by the rigid lid next step — the _ab2_step routing).  h_u/h_v
        #   from Nnn.eta (O(dη) vs NEMO's Nbb geometry, residual #1). TRACERS take
        #   the full increment.
        if diss_incr_bb is not None:
            dT_diss_bb, dS_diss_bb, du_diss_bb, dv_diss_bb = diss_incr_bb
            du_diss_bc, _ = _split(du_diss_bb, h_u)
            dv_diss_bc, _ = _split(dv_diss_bb, h_v)
        else:
            dT_diss_bb = dS_diss_bb = 0.0
            du_diss_bc = dv_diss_bc = 0.0
        # baroclinic leap-frog: u'(Nbb) + (u'_expl - u'(Nnn)) + 2dt·diss'(Nbb)
        u_naa = ((ubc_bef + (ubc_exp - ubc_now)) + du_diss_bc + btu_exp) * u_mask3
        u_naa = u_naa.at[:, -1].set(u_naa[:, 0])          # periodic-lon wrap
        v_naa = ((vbc_bef + (vbc_exp - vbc_now)) + dv_diss_bc + btv_exp) * v_mask3
        # Tracers: CARRY closed/dry cells (hold the now value) instead of
        # pinning to 0.  Pinning at 0 is correct for velocities (walls) but
        # on a 3-D staircase (full_step/partial cells) it writes T=0 into
        # every below-bottom cell — the #480 masked-cold-cell poison — which
        # staircase-adjacent stencils then pull into wet bottom cells (rest
        # state explodes in ~30 steps; homogeneous-T rest test catches it).
        T_naa = jnp.where(
            mask3 > 0,
            state.T_before.data + (state_expl.T.data - state.T.data) + dT_diss_bb,
            state.T.data)
        S_naa = jnp.where(
            mask3 > 0,
            state.S_before.data + (state_expl.S.data - state.S.data) + dS_diss_bb,
            state.S.data)
        _nemo_tracer_content_rhs = None
        _combine = getattr(_cfg_b, "tracer_combine", "concentration")
        if _combine not in ("concentration", "thickness_weighted"):
            raise ValueError(
                f"unknown tracer_combine {_combine!r}; expected "
                '"concentration" or "thickness_weighted"')
        if _combine == "thickness_weighted":
            # NEMO trazdf.F90:271-278 —
            #     e3t(Kaa)·T(Kaa) = e3t(Kbb)·T(Kbb) + 2·rdt·e3t(Kmm)·RHS
            # Combine CONTENT, not concentration.  The bare-concentration form
            # above does not conserve tracer content under a moving (z-star)
            # coordinate: +8.6e-6 drift in globally-integrated heat over 200
            # forcing-free steps vs NEMO's +3.4e-16 (#1226).
            #
            # Mapping onto this routine's intermediates: ``_step_impl`` already
            # builds the flux-form content update (hT = h_now·T_now − dt·div,
            # then T_expl = hT / h_naa), so the RAW advective content increment
            # is exactly
            #     h_naa·T_expl − h_now·T_now        ( = −2·rdt·div )
            # with no reweighting — matching NEMO's 2·rdt·e3t(Kmm)·RHS_adv,
            # whose trends are each divided by e3t(Kmm) before being multiplied
            # by it again.  The dissipative increment arrives as a CONCENTRATION
            # tendency from the Nbb pass, so it takes the Kmm ("now") thickness
            # — the weight NEMO gives every trend in ts(:,:,:,:,Nrhs), including
            # tra_ldf, which it evaluates at Kbb but still weights by e3t(Kmm).
            # ``h_k`` (computed above) is the Nnn thickness; eta_naa comes from
            # the barotropic solve, so h at state_expl.eta is the Kaa thickness.
            h_bef = compute_layer_thickness(
                state.eta_before.data, state.H_bathy.data, _zc,
                min_water_column_m=_cfg_b.min_water_column_m)
            h_naa = compute_layer_thickness(
                state_expl.eta.data, state.H_bathy.data, _zc,
                min_water_column_m=_cfg_b.min_water_column_m)
            _content_t = thickness_weighted_tracer_content(
                state.T_before.data, state.T.data, state_expl.T.data,
                dT_diss_bb, h_bef, h_k, h_naa)
            _content_s = thickness_weighted_tracer_content(
                state.S_before.data, state.S.data, state_expl.S.data,
                dS_diss_bb, h_bef, h_k, h_naa)
            T_naa = thickness_weighted_tracer_combine(
                state.T_before.data, state.T.data, state_expl.T.data,
                dT_diss_bb, h_bef, h_k, h_naa, mask3)
            S_naa = thickness_weighted_tracer_combine(
                state.S_before.data, state.S.data, state_expl.S.data,
                dS_diss_bb, h_bef, h_k, h_naa, mask3)
            _nemo_tracer_content_rhs = (_content_t, _content_s)
        eta_naa = state_expl.eta.data * cmask   # from the barotropic solve
        naa_expl = state_expl._replace(
            u=state_expl.u.replace(data=u_naa),
            v=state_expl.v.replace(data=v_naa),
            T=state_expl.T.replace(data=T_naa),
            S=state_expl.S.replace(data=S_naa),
            eta=state_expl.eta.replace(data=eta_naa),
        )

        # 3. Implicit vertical friction/diffusion ONCE, backward-Euler over
        #    rDt=2dt on the after-state (NEMO dyn_zdf/tra_zdf, rDt=2dt).
        dt_mom = rdt / _cfg_b.dt_mom_ratio
        _tke_prog = self._tke_prognostic_active()
        _tke_old = (state.tke.data if (_tke_prog and state.tke is not None)
                    else None)
        tke_new = None
        if _cfg_b.implicit_vertical_mixing:
            _res = self._apply_implicit_vertical_mixing(
                naa_expl, rdt, surface_forcing,
                K_v_phys=K_v_phys, A_v_phys=A_v_phys, K33_iso=k33_implicit,
                dt_mom=dt_mom,
                # zdftke advances en with the base rn_Dt even though the
                # surrounding MLF tracer/momentum implicit solves use
                # rDt=2*rn_Dt (dom_oce.F90:69-71; zdftke.F90:336-337).
                # Only nemo_literal consumes this override; every factored
                # TKE/CATKE/non-TKE card remains on dt_mom byte-for-byte.
                tke_rn_dt=dt,
                surface_tracer_forcing=surface_tracer_forcing,
                tracer_source=tracer_source,
                tke_old=_tke_old, tke_source=tke_source,
                return_tke=_tke_prog, grid=_grid,
                n2_tracers=self._n2_before_advection_tracers(state, z_coord=z_coord, config=config),
                n2_tracers_before=self._n2_nemo_before_tracers(state, z_coord=z_coord, config=config),
                tke_n2_bundle=_tke_n2_bundle,
                # NEMO e3w(Kmm) divisor (trazdf.F90:219-221): naa_expl.eta is
                # the barotropic AFTER/Kaa level ("h at state_expl.eta is the
                # Kaa thickness" above); state.eta is the Nnn/NOW level (the
                # same eta h_k is built from). This is THE production path for
                # the DINO kamm_mlf recipe (outer_integrator=leapfrog). Read
                # only under the NEMO identity
                # (zdf_implicit_solver_evaluation="nemo_literal").
                eta_now=state.eta.data,
                u_now=state.u.data, v_now=state.v.data,
                nemo_tracer_content_rhs=_nemo_tracer_content_rhs,
            z_coord=z_coord, config=config, iwm_fields=iwm_fields)
            if _tke_prog:
                naa, tke_new = _res
            else:
                naa = _res
        else:
            naa = naa_expl
        if tke_new is not None:
            if self._tke_advection_active():
                tke_new, _dtke_field = self._apply_tke_advection(
                    state, tke_new, rdt, grid=_grid, z_coord=z_coord, config=config)
                naa = naa._replace(dtke=_dtke_field)
            naa = naa._replace(
                tke=Field(data=tke_new, name="tke",
                          dims=("lat", "lon", "level"), units="m^2/s^2"))

        # 3b. NEMO ``mlf_baro_corr`` (see ``_apply_after_level_reconcile``),
        #     at NEMO's position: after the implicit vertical solve above and
        #     before the Asselin filter below. ONE implementation, called from
        #     both outer-step paths. "off" (the default) is bit-identical.
        naa = self._apply_after_level_reconcile(
            naa, state, btu_exp, btv_exp, u_mask3, v_mask3, _grid,
            kaa_eta_raw=kaa_eta_raw, z_coord=z_coord, config=config)

        # 4. Conservation fixer on the final after-state.
        if _cfg_b.use_conservation_fixer:
            naa = ocean_conservation_fixer(
                naa, state, _grid, _zc, _cfg_b)

        # 5. Robert-Asselin filter on the NOW fields (plain RA — DINO, not
        #    Williams); carried as the NEXT step's before-state Nbb.
        #    MOMENTUM (u,v): PLAIN velocity filter — DINO runs ln_dynadv_vec=.TRUE.
        #    (vector-form advection), for which NEMO dynatf_qco.F90:151-155 applies
        #    the Asselin filter to the raw velocity (NOT thickness-weighted; the
        #    thickness-weighted momentum branch :158-169 is the flux-form
        #    nn_dynkeg path, unused by DINO).  SSH (eta): PLAIN filter (ssh_atf).
        #    TRACERS (T,S): THICKNESS-WEIGHTED content filter (tra_atf_qco_lf,
        #    key_qco z*) — filter (e3t·T) then divide by the filtered thickness so
        #    heat/salt content is conserved under the moving z* coordinate
        #    (residual #2 close).
        def _asselin(now, before, after, m):
            return (now + gamma * (before - 2.0 * now + after)) * m
        kmm_u, kmm_v = state.u.data, state.v.data
        _raw_cycle_refs = (
            "nemo_e3t_0", "nemo_hu_0", "nemo_hv_0", "nemo_e1e2t",
            "nemo_e1e2u", "nemo_e1e2v",
        )
        _kmm_cycle = (
            _cfg_b.barotropic.barotropic_after_reconcile
            == "nemo_mlf_baro_corr"
            and all(getattr(_zc, name, None) is not None
                    for name in _raw_cycle_refs))
        if _kmm_cycle:
            _, _, kmm_u, kmm_v = nemo_qco_kmm_velocity_cycle(
                state.eta.data, state.u.data, state.v.data,
                kaa_hu_avg, kaa_hv_avg, _zc, u_mask3, v_mask3, _grid)
        if _euler_start:
            # NEMO skips ALL THREE Asselin filters on the l_1st_euler step --
            # ssh (sshwzv.f90:443), tracers (traatf_qco.f90:152) and momentum
            # (dynatf_qco.f90:164), each by its own internal guard -- so the
            # next step's before level is the UNFILTERED now level.
            #
            # Momentum is not simply ``state.u`` though: ``mlf_baro_corr``'s
            # second block (stpmlf.f90:716-723, taken because DINO sets
            # ln_bt_fw=.FALSE.) rewrites puu(Kmm) BEFORE the index swap, and
            # that block carries no filter guard either.  ``kmm_u``/``kmm_v``
            # above already hold exactly that value -- the install at
            # dynspg_ts.f90:1003 undone at stpmlf.f90:720, a pair that cancels
            # algebraically and not bitwise.  The ``* umask`` is NEMO's own,
            # on the same statement.
            # ``kmm_u``/``kmm_v`` are the now-level velocity as NEMO leaves
            # it -- the mlf_baro_corr Kmm round trip when this card runs it,
            # the untouched now level otherwise.  The mask and the periodic
            # wrap apply either way: NEMO's own statement carries ``* umask``
            # (stpmlf.f90:720-721) and its lbc_lnk closes the zonal seam, and
            # the leap-frog arm below does both.  The early return this
            # replaced did NEITHER, which let a dry-face value survive into
            # the next step's depth mean -- inert from rest, a leak from any
            # other no-history state (review finding).
            u_f = kmm_u * u_mask3
            u_f = u_f.at[:, -1].set(u_f[:, 0])
            v_f = kmm_v * v_mask3
            eta_f = state.eta.data
            T_f, S_f = state.T.data, state.S.data
        else:
            u_f = _asselin(kmm_u, state.u_before.data, naa.u.data, u_mask3)
            u_f = u_f.at[:, -1].set(u_f[:, 0])
            v_f = _asselin(kmm_v, state.v_before.data, naa.v.data, v_mask3)
            eta_f = _asselin(state.eta.data, state.eta_before.data,
                             naa.eta.data, cmask)
            # Thickness at the three tracer time levels + the Asselin-filtered ssh.
            # e3t_now == h_k (already computed from state.eta above).  e3t_f is built
            # from the filtered eta (eta_f == NEMO r3t_f) so numerator & denominator
            # stay consistent (dynatf/tra_atf use the ssh_atf-filtered scale factor).
            _mwc = _cfg_b.min_water_column_m
            e3t_now = h_k
            e3t_bef = compute_layer_thickness(
                state.eta_before.data, state.H_bathy.data, _zc,
                min_water_column_m=_mwc)
            e3t_aft = compute_layer_thickness(
                naa.eta.data, state.H_bathy.data, _zc,
                min_water_column_m=_mwc)
            e3t_flt = compute_layer_thickness(
                eta_f, state.H_bathy.data, _zc, min_water_column_m=_mwc)
            T_f = _thickness_weighted_asselin(
                state.T.data, state.T_before.data, naa.T.data,
                e3t_now, e3t_bef, e3t_aft, e3t_flt, gamma, mask3)
            S_f = _thickness_weighted_asselin(
                state.S.data, state.S_before.data, naa.S.data,
                e3t_now, e3t_bef, e3t_aft, e3t_flt, gamma, mask3)
        naa = naa._replace(
            u_before=state.u.replace(data=u_f),
            v_before=state.v.replace(data=v_f),
            T_before=state.T.replace(data=T_f),
            S_before=state.S.replace(data=S_f),
            eta_before=state.eta.replace(data=eta_f),
        )
        # barotropic_forcing_centred (#1226 item 3): swap THIS step's
        # now-forcing into the carry for the NEXT step's before-value —
        # mirrors NEMO's sbcmod.F90:382-386 ``utau_b(:,:) = utauU(:,:)``
        # end-of-step swap.  CORRECTED (#1729): this used to say nit000 was
        # "handled separately" by a forward-Euler-start branch above. That
        # branch is gone; the Euler start falls through to here and seeds the
        # carry on the same line as every other step, which is what NEMO does
        # (the swap is unconditional at sbcmod.F90:382-386).
        if getattr(_cfg_b, "barotropic_forcing_centred", False):
            naa = naa._replace(
                **_seed_centred_forcing_carry(
                    surface_forcing, freshwater, _cfg_b.rho_0,
                    state.land_mask.data))
        return naa

    def _nemo_mlf_step(self, state: LatLonCGridOceanState, dt: float,
                       freshwater=None, surface_forcing=None, sponge=None,
                       *, grid=None, vertex_mask=None,
                       t_seconds=None,
                       external_tracer_rate=None, z_coord=None, config=None, iwm_fields=None) -> LatLonCGridOceanState:
        """SINGLE-PASS transcription of ``stpmlf.F90``'s ``stp_MLF`` (P1,
        ``docs/ocean/fidelity/nemo_mlf_step_transcription_spec.md`` §6 risk
        register item 4).  PRIVATE, NOT wired to ``outer_integrator`` (that is
        P2) — call directly for the verification-ladder rung (a) bit-comparison
        against ``_leapfrog_step``.

        ``_leapfrog_step`` achieves "``dyn_ldf``/``tra_ldf`` evaluated at Nbb"
        (NEMO's forward-in-time explicit lateral diffusion, unconditionally
        required because a LEAP-FROG-CENTRED diffusion term is unconditionally
        unstable) by running the ENTIRE tendency pipeline TWICE per step — once
        at Nnn for advection (``_ab2_scope_override="advective"`` withholds
        dissipation), once at Nbb keeping ONLY the withheld dissipative
        increment.  NEMO does this in ONE pass: ``dyn_ldf(Kbb,Kmm,...)``
        (stpmlf.F90:275) / ``tra_ldf(Kbb,Kmm,...)`` (stpmlf.F90:437) read the
        BEFORE-level velocity/tracer as their OWN call argument while every
        OTHER term in the SAME pass (``dyn_adv``, ``dyn_vor`` EEN, ``dyn_hpg``,
        ``dyn_spg_ts``, GM bolus transport, physics, forcing) reads Nnn.  This
        method reproduces that: ONE ``_step_impl`` call, with
        ``_ldf_state=(T_before, S_before, u_before, v_before)`` so ONLY the
        lateral-friction / lateral-diffusion / isoneutral-Redi calls (routed
        through ``latlon_cgrid_ocean_baroclinic_tendencies``'s ``ldf_state``
        kwarg, and the GM/Redi block in ``_step_impl`` — see their docstrings)
        read Nbb; the withheld dissipative increment (``diss_incr``, the SAME
        "advective"-scope mechanism ``_leapfrog_step`` already uses) is then
        the Nbb-evaluated dissipation, applied forward at weight 1.0 exactly
        as ``_leapfrog_step`` does.  Everything else in this method (the
        explicit combine, implicit vertical mixing, conservation fixer, Robert-
        Asselin filter, first-step Euler start, thickness-weighted tracer
        Asselin, barotropic-before seeding, FCT Kbb base, centred-forcing
        carry) is IDENTICAL to ``_leapfrog_step`` — this method changes ONLY
        the composition mechanism for the dissipative terms (one pass instead
        of two), per spec risk register item 4's "TRANSCRIBE THE PHYSICS,
        REPLACE THE MECHANISM" instruction.  Every other residual fix in
        ``_leapfrog_step``'s docstring (#1 barotropic-before-seed, #1b boxcar
        filter, #2 thickness-weighted Asselin, #3 live barotropic Coriolis,
        #5 barotropic substep scale) is carried forward UNCHANGED.

        finalize_lbc commit point (spec §6-1, resolved decision 3): NEMO calls
        ``lbc_lnk`` (stpmlf.F90:458, row 31) as an explicit halo/sign-
        convention commit AFTER ``dyn_zdf``/``tra_zdf`` and BEFORE the Asselin
        filters (rows 32-34).  legoESM has no discrete halo-exchange step here
        (single-array closed lat-lon domain, not MPI-sharded) — masking is
        applied CONTINUOUSLY at each stage as ``x * mask`` (every ``u_naa``/
        ``T_naa`` write below).  This is PROVABLY idempotent: for a boolean
        mask ``m`` (0/1), ``(x*m)*m == x*m`` for any finite ``x`` — reapplying
        the mask at a later "commit point" cannot change an already-masked
        array.  PROBED (not just asserted): ``scripts/tmp/_probe_finalize_lbc_
        idempotence.py`` compares this method's output against inserting an
        EXTRA no-op ``* land_mask``/``* u_mask``/``* v_mask`` reapplication
        immediately after the implicit-vmix stage (mimicking a discrete
        finalize_lbc commit point) — CONFIRMED bit-identical (max abs diff
        0.0 on T/S/u/v/eta over a 3-level rest-plus-front channel, 2 steps).
        No explicit finalize_lbc-equivalent call is needed.

        ``mlf_baro_corr`` (row 30) — BUILT, and the old waiver is RETRACTED
        (2026-08-21, #1455).  The waiver claimed this call was "a provable
        no-op ... no depth-mean source exists under
        ``surface_stress_implicit=False``".  Measurement refutes it: the
        implicit vertical solve IS a depth-mean source here, through the
        weighting difference between its own strip/re-add and the combine's
        split, worth +0.27 m3/s2 per southern u-row on the 90-day DINO twin.
        The call now runs via ``_apply_after_level_reconcile`` whenever
        ``BarotropicConfig.barotropic_after_reconcile="nemo_mlf_baro_corr"``,
        at NEMO's own position (after ``dyn_zdf``, before the Asselin filter).
        It stays OPT-IN, so a card that does not select it is unchanged.

        Implicit-vmix divisor (spec §6-4, resolved decision 4): the divisor
        is no longer a separate flag — it belongs to the NEMO identity
        ``zdf_implicit_solver_evaluation="nemo_literal"``, which
        ``outer_integrator="nemo_mlf"`` hard-requires at construction.  This
        method simply uses that divisor via the SAME ``eta_now=`` kwarg
        ``_leapfrog_step`` already threads to
        ``_apply_implicit_vertical_mixing``.
        """
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        from legoesm.ocean.state import Field
        _grid = grid if grid is not None else self.grid

        # --- FIRST step: NEMO's l_1st_euler start.  Handled exactly as in
        #     ``_leapfrog_step`` -- read that method's block for the full
        #     citation.  ``stp_MLF`` has no first-step variant: it sets
        #     rDt = rn_Dt (stpmlf.f90:131-133) and runs every statement,
        #     ``mlf_baro_corr`` (:534, guarded on ln_dynspg_ts alone)
        #     included; only the three Asselin filters step aside, each by its
        #     own internal guard.  So the start is a PARAMETERISATION of the
        #     body below -- rDt, the barotropic window scale, and whether the
        #     filters run -- not the early return this replaced (#1729).
        _euler_start = state.u_before is None
        if _euler_start:
            state = state._replace(
                u_before=state.u, v_before=state.v, T_before=state.T,
                S_before=state.S, eta_before=state.eta,
            )

        # --- LEAP-FROG + Asselin.
        rdt = (1.0 if _euler_start else 2.0) * dt
        gamma = _cfg_b.asselin_gamma
        _tke_n2_bundle = self._tke_step_entry_n2_bundle(
            state, z_coord=_zc, config=_cfg_b)
        u_mask3 = state.u_mask.data[..., jnp.newaxis]
        v_mask3 = state.v_mask.data[..., jnp.newaxis]
        cmask = state.land_mask.data
        mask3 = cmask[..., jnp.newaxis]
        if isinstance(_zc, OceanPartialCellCoordinate):
            _au3, _av3 = compute_face_masks_3d(_zc.is_active, _grid)
            u_mask3 = u_mask3 * _au3.astype(u_mask3.dtype)
            v_mask3 = v_mask3 * _av3.astype(v_mask3.dtype)
            mask3 = mask3 * _zc.is_active.astype(mask3.dtype)

        # 1. SINGLE tendency pass at Nnn, with dyn_ldf/tra_ldf (+ the isoneutral
        #    Redi tendency, GM/Redi's lego home for row 28) reading Nbb via
        #    ``_ldf_state`` -- replacing ``_leapfrog_step``'s TWO ``_step_impl``
        #    calls (one Nnn-advective, one whole-Nbb-pass-for-diss) with ONE.
        #    "advective" scope still withholds the dissipative increment onto
        #    ``diss_incr`` (same mechanism ``_leapfrog_step`` uses); the
        #    difference is WHAT that increment is evaluated on -- now the Nbb
        #    tracers/velocity read as a LOCAL ARGUMENT by dyn_ldf/tra_ldf only,
        #    not a second whole-state pass (stpmlf.F90:275/437).
        # rDt scale: 1 on the Euler start (rDt=dt), 2 on a leap-frog step.
        # On the Euler start this ALSO reproduces NEMO's own averaging window:
        # ``ll_fw_start`` is forced .TRUE. on the l_1st_euler step
        # (dynspg_ts.f90:230-232), moving the boxcar centre from 2*nn_e to
        # nn_e. VERIFIED against NEMO's ts_wgt at DINO's resolved nn_e=23:
        # weights agree to 0.0, transport weights to 1.4e-17, both windows.
        _baro_scale = 1 if _euler_start else 2
        state_expl, (K_v_phys, A_v_phys, k33_implicit, surface_tracer_forcing,
                     tke_source, diss_incr, tracer_source,
                     kaa_eta_raw, kaa_hu_avg, kaa_hv_avg) = self._step_impl(
            state, rdt, freshwater=freshwater, surface_forcing=surface_forcing,
            sponge=sponge, _apply_implicit_vmix=False, grid=_grid,
            vertex_mask=vertex_mask, t_seconds=t_seconds,
            _ab2_scope_override="advective",
            _barotropic_substep_scale=_baro_scale,
            _barotropic_before_state=(
                state.eta_before.data, state.u_before.data,
                state.v_before.data),
            # FCT/Zalesak base = Nbb (risk register item 2): unchanged by the
            # single-pass merge -- still threaded on the ONE pass that now
            # does both advection AND (via _ldf_state) dissipation.
            _fct_tracer_before=(
                state.T_before.data, state.S_before.data),
            _external_tracer_rate=external_tracer_rate,
            _ldf_state=(
                state.T_before.data, state.S_before.data,
                state.u_before.data, state.v_before.data),
            _tke_n2_bundle_override=_tke_n2_bundle,
            _return_raw_kaa_qco=True,
        z_coord=z_coord, config=config, iwm_fields=iwm_fields)

        # 2. Explicit combine -- IDENTICAL algebra to ``_leapfrog_step`` (same
        #    baroclinic/barotropic momentum split, same tracer combine, same
        #    thickness_weighted option); ``diss_incr`` now carries the
        #    single-pass Nbb-evaluated dissipation instead of a second pass's.
        h_k = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, _zc,
            min_water_column_m=_cfg_b.min_water_column_m)
        h_u = min_cell_to_uface(h_k)
        h_v = min_cell_to_vface(h_k, _grid)

        def _split(field, h_face):
            bt = depth_mean(field, h_face, 1.0e-10, keepdims=True, fused=False)
            return field - bt, bt      # (baroclinic deviation, barotropic mean)

        ubc_now, _ = _split(state.u.data, h_u)
        ubc_bef, _ = _split(state.u_before.data, h_u)
        ubc_exp, btu_exp = _split(state_expl.u.data, h_u)
        vbc_now, _ = _split(state.v.data, h_v)
        vbc_bef, _ = _split(state.v_before.data, h_v)
        vbc_exp, btv_exp = _split(state_expl.v.data, h_v)
        if diss_incr is not None:
            dT_diss_bb, dS_diss_bb, du_diss_bb, dv_diss_bb = diss_incr
            du_diss_bc, _ = _split(du_diss_bb, h_u)
            dv_diss_bc, _ = _split(dv_diss_bb, h_v)
        else:
            dT_diss_bb = dS_diss_bb = 0.0
            du_diss_bc = dv_diss_bc = 0.0
        u_naa = ((ubc_bef + (ubc_exp - ubc_now)) + du_diss_bc + btu_exp) * u_mask3
        u_naa = u_naa.at[:, -1].set(u_naa[:, 0])          # periodic-lon wrap
        v_naa = ((vbc_bef + (vbc_exp - vbc_now)) + dv_diss_bc + btv_exp) * v_mask3
        T_naa = jnp.where(
            mask3 > 0,
            state.T_before.data + (state_expl.T.data - state.T.data) + dT_diss_bb,
            state.T.data)
        S_naa = jnp.where(
            mask3 > 0,
            state.S_before.data + (state_expl.S.data - state.S.data) + dS_diss_bb,
            state.S.data)
        _combine = getattr(_cfg_b, "tracer_combine", "concentration")
        if _combine not in ("concentration", "thickness_weighted"):
            raise ValueError(
                f"unknown tracer_combine {_combine!r}; expected "
                '"concentration" or "thickness_weighted"')
        if _combine == "thickness_weighted":
            h_bef = compute_layer_thickness(
                state.eta_before.data, state.H_bathy.data, _zc,
                min_water_column_m=_cfg_b.min_water_column_m)
            h_naa = compute_layer_thickness(
                state_expl.eta.data, state.H_bathy.data, _zc,
                min_water_column_m=_cfg_b.min_water_column_m)
            T_naa = thickness_weighted_tracer_combine(
                state.T_before.data, state.T.data, state_expl.T.data,
                dT_diss_bb, h_bef, h_k, h_naa, mask3)
            S_naa = thickness_weighted_tracer_combine(
                state.S_before.data, state.S.data, state_expl.S.data,
                dS_diss_bb, h_bef, h_k, h_naa, mask3)
        eta_naa = state_expl.eta.data * cmask   # from the barotropic solve
        naa_expl = state_expl._replace(
            u=state_expl.u.replace(data=u_naa),
            v=state_expl.v.replace(data=v_naa),
            T=state_expl.T.replace(data=T_naa),
            S=state_expl.S.replace(data=S_naa),
            eta=state_expl.eta.replace(data=eta_naa),
        )

        # 3. Implicit vertical friction/diffusion ONCE, backward-Euler over
        #    rDt=2dt (NEMO dyn_zdf/tra_zdf, rows 22/29) -- identical to
        #    ``_leapfrog_step``.
        dt_mom = rdt / _cfg_b.dt_mom_ratio
        _tke_prog = self._tke_prognostic_active()
        _tke_old = (state.tke.data if (_tke_prog and state.tke is not None)
                    else None)
        tke_new = None
        if _cfg_b.implicit_vertical_mixing:
            _res = self._apply_implicit_vertical_mixing(
                naa_expl, rdt, surface_forcing,
                K_v_phys=K_v_phys, A_v_phys=A_v_phys, K33_iso=k33_implicit,
                dt_mom=dt_mom,
                tke_rn_dt=dt,
                surface_tracer_forcing=surface_tracer_forcing,
                tracer_source=tracer_source,
                tke_old=_tke_old, tke_source=tke_source,
                return_tke=_tke_prog, grid=_grid,
                n2_tracers=self._n2_before_advection_tracers(state, z_coord=z_coord, config=config),
                n2_tracers_before=self._n2_nemo_before_tracers(state, z_coord=z_coord, config=config),
                tke_n2_bundle=_tke_n2_bundle,
                eta_now=state.eta.data,
                u_now=state.u.data, v_now=state.v.data,
            z_coord=z_coord, config=config, iwm_fields=iwm_fields)
            if _tke_prog:
                naa, tke_new = _res
            else:
                naa = _res
        else:
            naa = naa_expl
        if tke_new is not None:
            if self._tke_advection_active():
                tke_new, _dtke_field = self._apply_tke_advection(
                    state, tke_new, rdt, grid=_grid, z_coord=z_coord, config=config)
                naa = naa._replace(dtke=_dtke_field)
            naa = naa._replace(
                tke=Field(data=tke_new, name="tke",
                          dims=("lat", "lon", "level"), units="m^2/s^2"))

        # 3b. NEMO ``mlf_baro_corr`` (see ``_apply_after_level_reconcile``),
        #     at NEMO's position: after the implicit vertical solve above and
        #     before the Asselin filter below. ONE implementation, called from
        #     both outer-step paths. "off" (the default) is bit-identical.
        naa = self._apply_after_level_reconcile(
            naa, state, btu_exp, btv_exp, u_mask3, v_mask3, _grid,
            kaa_eta_raw=kaa_eta_raw, z_coord=z_coord, config=config)

        # 4. Conservation fixer on the final after-state.
        if _cfg_b.use_conservation_fixer:
            naa = ocean_conservation_fixer(
                naa, state, _grid, _zc, _cfg_b)

        # 5. Robert-Asselin filter -- identical to ``_leapfrog_step``. This IS
        #    row 31's finalize_lbc position relative to rows 32-34 (Asselin):
        #    the continuous masking above already committed every write, so no
        #    extra step is inserted here (see the finalize_lbc docstring note).
        def _asselin(now, before, after, m):
            return (now + gamma * (before - 2.0 * now + after)) * m
        kmm_u, kmm_v = state.u.data, state.v.data
        _raw_cycle_refs = (
            "nemo_e3t_0", "nemo_hu_0", "nemo_hv_0", "nemo_e1e2t",
            "nemo_e1e2u", "nemo_e1e2v",
        )
        _kmm_cycle = (
            _cfg_b.barotropic.barotropic_after_reconcile
            == "nemo_mlf_baro_corr"
            and all(getattr(_zc, name, None) is not None
                    for name in _raw_cycle_refs))
        if _kmm_cycle:
            _, _, kmm_u, kmm_v = nemo_qco_kmm_velocity_cycle(
                state.eta.data, state.u.data, state.v.data,
                kaa_hu_avg, kaa_hv_avg, _zc, u_mask3, v_mask3, _grid)
        if _euler_start:
            # NEMO skips ALL THREE Asselin filters on the l_1st_euler step --
            # ssh (sshwzv.f90:443), tracers (traatf_qco.f90:152) and momentum
            # (dynatf_qco.f90:164), each by its own internal guard -- so the
            # next step's before level is the UNFILTERED now level.
            #
            # Momentum is not simply ``state.u`` though: ``mlf_baro_corr``'s
            # second block (stpmlf.f90:716-723, taken because DINO sets
            # ln_bt_fw=.FALSE.) rewrites puu(Kmm) BEFORE the index swap, and
            # that block carries no filter guard either.  ``kmm_u``/``kmm_v``
            # above already hold exactly that value -- the install at
            # dynspg_ts.f90:1003 undone at stpmlf.f90:720, a pair that cancels
            # algebraically and not bitwise.  The ``* umask`` is NEMO's own,
            # on the same statement.
            # ``kmm_u``/``kmm_v`` are the now-level velocity as NEMO leaves
            # it -- the mlf_baro_corr Kmm round trip when this card runs it,
            # the untouched now level otherwise.  The mask and the periodic
            # wrap apply either way: NEMO's own statement carries ``* umask``
            # (stpmlf.f90:720-721) and its lbc_lnk closes the zonal seam, and
            # the leap-frog arm below does both.  The early return this
            # replaced did NEITHER, which let a dry-face value survive into
            # the next step's depth mean -- inert from rest, a leak from any
            # other no-history state (review finding).
            u_f = kmm_u * u_mask3
            u_f = u_f.at[:, -1].set(u_f[:, 0])
            v_f = kmm_v * v_mask3
            eta_f = state.eta.data
            T_f, S_f = state.T.data, state.S.data
        else:
            u_f = _asselin(kmm_u, state.u_before.data, naa.u.data, u_mask3)
            u_f = u_f.at[:, -1].set(u_f[:, 0])
            v_f = _asselin(kmm_v, state.v_before.data, naa.v.data, v_mask3)
            eta_f = _asselin(state.eta.data, state.eta_before.data,
                             naa.eta.data, cmask)
            _mwc = _cfg_b.min_water_column_m
            e3t_now = h_k
            e3t_bef = compute_layer_thickness(
                state.eta_before.data, state.H_bathy.data, _zc,
                min_water_column_m=_mwc)
            e3t_aft = compute_layer_thickness(
                naa.eta.data, state.H_bathy.data, _zc,
                min_water_column_m=_mwc)
            e3t_flt = compute_layer_thickness(
                eta_f, state.H_bathy.data, _zc, min_water_column_m=_mwc)
            T_f = _thickness_weighted_asselin(
                state.T.data, state.T_before.data, naa.T.data,
                e3t_now, e3t_bef, e3t_aft, e3t_flt, gamma, mask3)
            S_f = _thickness_weighted_asselin(
                state.S.data, state.S_before.data, naa.S.data,
                e3t_now, e3t_bef, e3t_aft, e3t_flt, gamma, mask3)
        naa = naa._replace(
            u_before=state.u.replace(data=u_f),
            v_before=state.v.replace(data=v_f),
            T_before=state.T.replace(data=T_f),
            S_before=state.S.replace(data=S_f),
            eta_before=state.eta.replace(data=eta_f),
        )
        if getattr(_cfg_b, "barotropic_forcing_centred", False):
            naa = naa._replace(
                **_seed_centred_forcing_carry(
                    surface_forcing, freshwater, _cfg_b.rho_0,
                    state.land_mask.data))
        return naa

    def _unsplit_ab2_step(self, state: LatLonCGridOceanState, dt: float,
                          freshwater=None, surface_forcing=None, sponge=None,
                          *, grid=None, vertex_mask=None, z_coord=None, config=None, iwm_fields=None) -> LatLonCGridOceanState:
        """MITgcm-faithful UNSPLIT implicit free-surface AB2 step (no mode split).

        Replaces the split-explicit barotropic/baroclinic stepping (which breaks
        the discrete PGF/continuity adjointness at the grid scale, driving the
        spurious 2dx baroclinic instability) with MITgcm's unsplit
        ``implicitFreeSurface`` algorithm (docs/ocean/fidelity/
        mitgcm_unsplit_freesurface_fix.md):

          1. u* = u^n + AB2(Δt·du_dt)   — the FULL explicit baroclinic tendency
             (advection + Coriolis + baroclinic hydrostatic PGF + lateral diss;
             planetary f×u is in du_dt under coriolis_scheme='explicit_ab2'), AB2-
             extrapolated on the FULL 3-D velocity (NO depth-mean / barotropic split).
          2. One implicit elliptic solve for eta^{n+1} from the depth-integrated
             divergence of the predictor transport (``solve_unsplit_freesurface``).
          3. u^{n+1} = u* − Δt·g·∇eta^{n+1}, the SAME surface-pressure gradient on
             every level (the surface PGF is implicit/backward-Euler).
          4. Implicit vertical mixing once (backward-Euler vert friction + diffusion
             + surface wind/tracer BC) — Veros/MITgcm split-implicit convention.

        Opt-in via ``barotropic_solver='implicit_unsplit'`` (requires
        ``outer_integrator='ab2'``); the split ``implicit_cn`` path is untouched.
        """
        _zc = self.z_coord if z_coord is None else z_coord  # SPMD band override
        _cfg_b = self.config if config is None else config  # SPMD band override
        from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
            solve_unsplit_freesurface,
        )
        from legoesm.ocean.state import Field
        if freshwater is not None:
            raise ValueError(
                'barotropic_solver="implicit_unsplit" does not yet support the '
                "freshwater argument (E-P-R free-surface forcing + virtual-salt "
                'flux); use barotropic_solver="implicit_cn".')
        _grid = grid if grid is not None else self.grid
        g = _cfg_b.g
        eps = _cfg_b.ab2_epsilon
        a_n, a_p = 1.5 + eps, 0.5 + eps
        u_mask = state.u_mask.data
        v_mask = state.v_mask.data
        cmask = state.land_mask.data
        u_mask3 = u_mask[..., jnp.newaxis]
        v_mask3 = v_mask[..., jnp.newaxis]
        mask3 = cmask[..., jnp.newaxis]
        if isinstance(_zc, OceanPartialCellCoordinate):
            _au3, _av3 = compute_face_masks_3d(_zc.is_active, _grid)
            u_mask3 = u_mask3 * _au3.astype(u_mask3.dtype)
            v_mask3 = v_mask3 * _av3.astype(v_mask3.dtype)
            mask3 = mask3 * _zc.is_active.astype(mask3.dtype)

        # 1. Explicit baroclinic tendency (NO surface PGF, NO implicit vmix).
        tend = self.tendencies(state, surface_forcing, sponge=sponge, dt=dt,
                               grid=_grid, vertex_mask=vertex_mask, z_coord=z_coord, config=config)
        du_n = dt * tend.du_dt.data
        dv_n = dt * tend.dv_dt.data
        dT_n = dt * tend.dT_dt.data    # noqa: N806 (T = temperature, domain convention)
        dS_n = dt * tend.dS_dt.data    # noqa: N806 (S = salinity)

        # GM/Redi isopycnal + skew (bolus) TRACER mixing — not in self.tendencies()
        # (it lives in _step_impl).  Evaluated at u^n and added to the explicit tracer
        # increment so it AB2-extrapolates with the rest (ab2_scope="total").  GM/Redi
        # is a tracer-only scheme (skew flux), so it does NOT touch the momentum.  The
        # vertical isoneutral diagonal K33 (when implicit_K33) is computed from the
        # same density/slopes and folded into the implicit vertical-mixing solve.
        k33_iso = None
        if _cfg_b.gm_redi is not None:
            gm_cfg = _cfg_b.gm_redi
            _kri_static, _kri_v_static = static_kappa_redi_override(gm_cfg, _grid)
            _eos_depth = getattr(_cfg_b, "eos_depth", "insitu")
            _gm_dj = None
            if gm_cfg.implicit_K33:
                _gm_dj = gm_redi_density_and_jacobian(
                    state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
                    _grid, _zc, eos=_cfg_b.eos,
                    eos_linear=_cfg_b.eos_linear,
                    eos_nemo_seos=_cfg_b.eos_nemo_seos, mask=cmask,
                    rho_0=_cfg_b.constants.rho_0, g=_cfg_b.constants.g,
                    eos_depth=_eos_depth)
            dT_gm, dS_gm = gm_redi_tracer_tendency_latlon(  # noqa: N806
                state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
                _grid, _zc, gm_cfg, eos=_cfg_b.eos,
                eos_linear=_cfg_b.eos_linear,
                eos_nemo_seos=_cfg_b.eos_nemo_seos, mask=cmask,
                u_mask=u_mask, v_mask=v_mask,
                rho_0=_cfg_b.constants.rho_0, g=_cfg_b.constants.g,
                omega=_cfg_b.omega,  # see the sibling call's comment above
                kappa_redi_override=_kri_static,
                kappa_redi_v_override=_kri_v_static,
                density_jacobian=_gm_dj,
                native_slope_eta=state.eta.data,
                redi_flux_eta=state.eta.data,
                dt=dt, eos_depth=_eos_depth)
            dT_n = dT_n + dt * dT_gm    # noqa: N806
            dS_n = dS_n + dt * dS_gm    # noqa: N806
            if gm_cfg.implicit_K33:
                k33_iso = compute_isoneutral_K33_latlon(
                    state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
                    _grid, _zc, gm_cfg, eos=_cfg_b.eos,
                    eos_linear=_cfg_b.eos_linear,
                    eos_nemo_seos=_cfg_b.eos_nemo_seos, mask=cmask,
                    rho_0=_cfg_b.constants.rho_0, g=_cfg_b.constants.g,
                    omega=_cfg_b.omega,
                    kappa_redi_override=_kri_static,
                    kappa_redi_v_override=_kri_v_static,
                    density_jacobian=_gm_dj,
                    native_slope_eta=state.eta.data,
                    # #1226: same wall masks as the tendency call above.
                    u_mask=u_mask, v_mask=v_mask, dt=dt, eos_depth=_eos_depth)
        du_p = (state.u_incr_prev.data if state.u_incr_prev is not None
                else jnp.zeros_like(du_n))
        dv_p = (state.v_incr_prev.data if state.v_incr_prev is not None
                else jnp.zeros_like(dv_n))
        dT_p = (state.T_incr_prev.data if state.T_incr_prev is not None  # noqa: N806
                else jnp.zeros_like(dT_n))
        dS_p = (state.S_incr_prev.data if state.S_incr_prev is not None  # noqa: N806
                else jnp.zeros_like(dS_n))
        # 2. AB2 predictor on the FULL 3-D velocity + tracers.
        # NB (#517 item 8): kept inline (NOT ab2_blend) — adding the blend into
        # ``state.X.data`` re-associates the FP add (~1e-16 drift); see the
        # split-predictor note above.
        u_star = (state.u.data + a_n * du_n - a_p * du_p) * u_mask3
        v_star = (state.v.data + a_n * dv_n - a_p * dv_p) * v_mask3
        T_star = (state.T.data + a_n * dT_n - a_p * dT_p) * mask3   # noqa: N806
        S_star = (state.S.data + a_n * dS_n - a_p * dS_p) * mask3   # noqa: N806

        # 3. UNSPLIT implicit free surface + uniform surface-pressure correction.
        h_k = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, _zc,
            min_water_column_m=_cfg_b.min_water_column_m)
        h_u = min_cell_to_uface(h_k)
        h_v = min_cell_to_vface(h_k, _grid)
        eta_new, u_corr, v_corr = solve_unsplit_freesurface(
            state.eta.data, u_star, v_star, h_u, h_v, dt, g, _grid,
            cmask, u_mask, v_mask)
        state_corr = state._replace(
            u=state.u.replace(data=u_corr),
            v=state.v.replace(data=v_corr),
            T=state.T.replace(data=T_star),
            S=state.S.replace(data=S_star),
            eta=state.eta.replace(data=eta_new))

        # 4. Implicit vertical mixing once (recomputes K_v/A_v from state_corr;
        #    applies the surface wind/tracer BC + restoring internally).
        state_new = self._apply_implicit_vertical_mixing(
            state_corr, dt, surface_forcing, K33_iso=k33_iso, grid=_grid,
            # NEMO eosbn2 Nnow N²: step-entry (before-advection) T/S.
            n2_tracers=self._n2_before_advection_tracers(state, z_coord=z_coord, config=config),
            # NEMO e3w(Kmm) divisor (trazdf.F90:219-221): the true
            # pre-barotropic-solve NOW eta (state_corr.eta is the AFTER/Naa
            # level built at step 3 above). Read only under the NEMO identity
            # (zdf_implicit_solver_evaluation="nemo_literal").
            eta_now=state.eta.data,
            u_now=state.u.data, v_now=state.v.data, z_coord=z_coord, config=config, iwm_fields=iwm_fields)

        # 5. Carry the explicit increments for the next AB2 step.
        return state_new._replace(
            T_incr_prev=Field(data=dT_n * mask3, name="T_incr_prev",
                              dims=state.T.dims, units=state.T.units),
            S_incr_prev=Field(data=dS_n * mask3, name="S_incr_prev",
                              dims=state.S.dims, units=state.S.units),
            u_incr_prev=Field(data=du_n * u_mask3, name="u_incr_prev",
                              dims=state.u.dims, units=state.u.units),
            v_incr_prev=Field(data=dv_n * v_mask3, name="v_incr_prev",
                              dims=state.v.dims, units=state.v.units),
        )

    def step_checked(
        self,
        state: LatLonCGridOceanState,
        dt: float,
        freshwater=None,
        surface_forcing=None,
        sponge=None,
        *,
        t_seconds=None,
    ) -> LatLonCGridOceanState:
        """Advance one timestep with host-side runtime validation."""
        if not self._cfl_checked:
            self.check_barotropic_cfl(dt)
            self._cfl_checked = True
        state_new = self.step(state, dt, freshwater=freshwater,
                              surface_forcing=surface_forcing,
                              sponge=sponge, t_seconds=t_seconds)
        if self.config.runtime_checks.enable_runtime_checks:
            self._assert_runtime_invariants(state_new)
        return state_new

    def _assert_runtime_invariants(self, state: LatLonCGridOceanState) -> None:
        """Host-side runtime checks for debugging/regression hardening."""
        mask = state.land_mask.data
        wet = mask > 0.5

        # Face mask consistency: u_mask/v_mask must match land_mask
        # (thread the geometry so a walled-but-wet partial-periodic seam
        # state — all cells wet, seam u-face closed — passes).
        u_expected, v_expected = compute_face_masks(mask, self.grid)
        if not (bool(jnp.all(state.u_mask.data == u_expected))
                and bool(jnp.all(state.v_mask.data == v_expected))):
            raise ValueError(
                "C-grid ocean: u_mask/v_mask inconsistent with land_mask. "
                "Use replace_land_mask() or land_mask_override instead of "
                "raw state._replace(land_mask=...).")

        # Fuse all reductions into a single ``jnp.stack`` + host pull
        # so the runtime check costs one device→host stall instead of
        # 7-9.  Same pattern as the loop-3 ``ocean_model.py`` rewrite.
        water_col = state.eta.data + state.H_bathy.data
        wet3 = wet[..., jnp.newaxis]
        T_ocean = jnp.where(wet3, state.T.data, jnp.nan)
        S_ocean = jnp.where(wet3, state.S.data, jnp.nan)

        finite_ok = (
            jnp.all(jnp.isfinite(state.u.data))
            & jnp.all(jnp.isfinite(state.v.data))
            & jnp.all(jnp.isfinite(state.T.data))
            & jnp.all(jnp.isfinite(state.S.data))
            & jnp.all(jnp.isfinite(state.eta.data))
        )
        any_wet = jnp.any(wet)
        _eta_dtype = state.eta.data.dtype
        _stats = jnp.stack([
            finite_ok.astype(_eta_dtype),
            any_wet.astype(_eta_dtype),
            jnp.min(jnp.where(wet, water_col, jnp.inf)).astype(_eta_dtype),
            jnp.max(jnp.abs(jnp.where(wet, state.eta.data, 0.0))).astype(_eta_dtype),
            jnp.nanmin(T_ocean).astype(_eta_dtype),
            jnp.nanmax(T_ocean).astype(_eta_dtype),
            jnp.nanmin(S_ocean).astype(_eta_dtype),
            jnp.nanmax(S_ocean).astype(_eta_dtype),
        ])
        host = np.asarray(_stats)
        finite_ok_h = bool(host[0] > 0.5)
        any_wet_h = bool(host[1] > 0.5)
        min_wc = float(host[2]) if any_wet_h else float("inf")
        eta_abs = float(host[3]) if any_wet_h else 0.0
        T_min = float(host[4]) if any_wet_h else float("nan")
        T_max = float(host[5]) if any_wet_h else float("nan")
        S_min = float(host[6]) if any_wet_h else float("nan")
        S_max = float(host[7]) if any_wet_h else float("nan")

        if not finite_ok_h:
            raise FloatingPointError(
                "C-grid ocean: non-finite state detected")

        if min_wc < self.config.min_water_column_m:
            raise ValueError(
                f"C-grid ocean: water column too small. "
                f"min(eta+H)={min_wc:.6g} m, "
                f"threshold={self.config.min_water_column_m:.6g} m",
            )

        if eta_abs > self.config.runtime_checks.max_abs_eta_m:
            raise ValueError(
                f"C-grid ocean: |eta|={eta_abs:.3g} exceeds "
                f"threshold {self.config.runtime_checks.max_abs_eta_m:.3g}",
            )

        if any_wet_h and (
            T_min < self.config.runtime_checks.temperature_min_c
            or T_max > self.config.runtime_checks.temperature_max_c
        ):
            raise ValueError(
                f"C-grid ocean: T range [{T_min:.3f}, {T_max:.3f}] "
                f"outside bounds [{self.config.runtime_checks.temperature_min_c:.3f}, "
                f"{self.config.runtime_checks.temperature_max_c:.3f}]",
            )

        if any_wet_h and (
            S_min < self.config.runtime_checks.salinity_min_psu
            or S_max > self.config.runtime_checks.salinity_max_psu
        ):
            raise ValueError(
                f"C-grid ocean: S range [{S_min:.3f}, {S_max:.3f}] "
                f"outside bounds [{self.config.runtime_checks.salinity_min_psu:.3f}, "
                f"{self.config.runtime_checks.salinity_max_psu:.3f}]",
            )

    def integrate(
        self,
        state: LatLonCGridOceanState,
        duration: float,
        dt: float,
        save_every: int = 1,
        t0_seconds: float = 0.0,
    ) -> tuple[LatLonCGridOceanState, list[LatLonCGridOceanState]]:
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : LatLonCGridOceanState
        duration : float
            Total integration time [seconds].
        dt : float
            Time step [seconds].
        save_every : int
            Save state every N steps.

        Returns
        -------
        final_state, trajectory
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt!r}")
        if duration < 0.0:
            raise ValueError(f"duration must be >= 0, got {duration!r}")
        if save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {save_every!r}")

        n_steps = int(duration / dt)
        if duration > 0.0 and n_steps < 1:
            raise ValueError(
                f"zero steps; increase duration or reduce dt "
                f"(duration={duration!r}, dt={dt!r})",
            )

        # store_mass_flux (#1442, codex round-6 YELLOW 5): seed BEFORE the
        # first step and before trajectory[0] is captured.  Unseeded, the
        # jitted step sees a different input treedef on iteration 2 (None ->
        # Field) and RETRACES, and trajectory[0] carries a different pytree
        # structure from every later entry -- so a caller that stacks the
        # trajectory gets a structure error rather than an array.  No-op when
        # the flag is off.
        state = seed_mass_flux_carry(
            state, getattr(self.config, "store_mass_flux", False))
        state = seed_salt_flux_carry(
            state, getattr(self.config, "store_salt_flux", False))

        trajectory = [state]
        step_fn = self.step_checked if self.config.runtime_checks.enable_runtime_checks else self.step
        # Thread the traced elapsed model time ONLY when the equilibrium tide is
        # on; otherwise call step_fn exactly as before (bit-identical). t is a
        # device array (NOT a Python float) so the jitted step is compiled once,
        # not retraced per step. t=0 <-> the constituents' reference epoch.
        _tf = getattr(self.config, "tidal_forcing", None)
        _tide_on = _tf is not None and _tf.enabled
        for i in range(n_steps):
            if _tide_on:
                state = step_fn(state, dt,
                                t_seconds=jnp.asarray(t0_seconds + i * dt))
            else:
                state = step_fn(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory

    def seed_scan_carry(self, state, dt, **step_kwargs):
        """Prepare a raw initial state into a CONSTANT-pytree, dtype-stable carry.

        Adds the AB2 / rigid-lid / EKE / TKE carry Fields the step writes (so the
        pytree treedef is FIXED across iterations — a ``None -> Field`` transition
        mid-scan crashes ``jax.lax.scan``, and a direct ``step`` loop would change
        its treedef on the first step) AND reconciles every leaf's dtype to the
        step's output dtype.

        DTYPE: the step promotes the prognostic dynamics to the working (compute)
        float precision — driven by the f64 vertical-coordinate geometry under
        x64 — while pinning EKE / ``eke_diss`` and the rigid-lid streamfunction
        back to the storage precision (an explicit ``convert_element_type``): a
        MIXED but stable fixed point (idempotent after one step). Seeding the
        carry at the working precision, then matching each leaf's dtype to what
        the step ACTUALLY outputs (discovered via ``jax.eval_shape`` — abstract,
        no FLOPs), makes ``carry-in == carry-out`` so the scan body's
        equal-types contract holds. Storage-pinned leaves (EKE / ``eke_diss`` /
        the rigid-lid streamfunction) are returned to storage precision exactly
        (their f32->f64->f32 round-trip is exact). This is NOT bit-identical to a
        raw direct step that begins from f32 storage inputs: the carry feeds the
        FIRST step its promoted leaves (T/u/v) at the working precision the step
        would compute in anyway, vs the raw IC's storage precision — an O(1e-7)
        difference in T (verified ~3.8e-7) that is the unavoidable, and more
        self-consistent, cost of a fixed-point scan carry. No physics /
        conservation change. ``work_dtype`` is SELF-TARGETING: on an all-f32
        build (x32 / f32 geometry) it is f32, so the whole reconciliation is a
        no-op and the prior behaviour is preserved bit-for-bit; it only promotes
        when the step would promote anyway (and the un-seeded scan would
        otherwise crash).

        ``step_kwargs`` (e.g. ``surface_forcing``) MUST match the kwargs of the
        steps that follow: surface forcing can change a tendency leaf's promoted
        dtype. ``integrate_scan`` passes none (its ``scan_fn`` calls
        ``step(state, dt)``); a forced direct-loop driver passes its forcing.
        Idempotent: re-seeding an already-prepared carry is a no-op.
        """
        # Prime build-once caches from the CONCRETE input state before
        # the scan traces step() with tracers (codex round-2 MINOR).
        self.prime_step_caches(state)
        # NEMO's carried after-SSH slot: the step writes it every iteration,
        # so an unseeded None->Field transition would crash the scan on the
        # first one.  The seed is NEMO's OWN value before any step has run --
        # ssh(:,:,Kaa) = ssh(:,:,Kbb) (restart.F90:370) -- so seeding changes
        # no number: the wzv branch reads the step-entry height either way.
        if (nemo_rk3_after_ssh_is_carried(self.config)
                and state.eta_rk3_after is None):
            state = state._replace(eta_rk3_after=state.eta)
        # Pre-initialize AB2 carry fields so the pytree structure
        # is stable across scan iterations (None → Field transition
        # would crash jax.lax.scan).
        if (self.config.tracer_time_integrator == "ab2"
                and state.T_flux_div_prev is None):
            from legoesm.core.field import Field
            _dims_fd = ("lat", "lon", "level")
            _zero = jnp.zeros_like(state.T.data)
            state = state._replace(
                T_flux_div_prev=Field(
                    data=_zero, name="T_flux_div_prev",
                    dims=_dims_fd, units="m/s"),
                S_flux_div_prev=Field(
                    data=_zero, name="S_flux_div_prev",
                    dims=_dims_fd, units="m/s"),
            )

        # AB2 outer integrator: seed the prior-increment carry to ZERO so the scan
        # keeps a constant pytree (None -> Field would crash lax.scan). The first
        # step then uses ΔX^{n-1}=0 (a 1.6× forward-Euler seed).
        if (getattr(self.config, "outer_integrator", "forward_euler") == "ab2"
                and state.T_incr_prev is None):
            from legoesm.core.field import Field
            _z3 = jnp.zeros_like(state.T.data)
            _zu = jnp.zeros_like(state.u.data)
            _zv = jnp.zeros_like(state.v.data)
            state = state._replace(
                T_incr_prev=Field(data=_z3, name="T_incr_prev",
                                  dims=state.T.dims, units=state.T.units),
                S_incr_prev=Field(data=jnp.zeros_like(state.S.data),
                                  name="S_incr_prev",
                                  dims=state.S.dims, units=state.S.units),
                u_incr_prev=Field(data=_zu, name="u_incr_prev",
                                  dims=state.u.dims, units=state.u.units),
                v_incr_prev=Field(data=_zv, name="v_incr_prev",
                                  dims=state.v.dims, units=state.v.units),
            )

        # Leapfrog-family outer integrator (NEMO stp_MLF: "leapfrog"
        # _leapfrog_step OR "nemo_mlf" _nemo_mlf_step, P2 -- both write the
        # SAME {u,v,T,S,eta}_before Fields every step, per _nemo_mlf_step's
        # docstring): seed the before-state Nbb Fields (= copies of the
        # now-fields) so the scan carry keeps a CONSTANT pytree -- the eager
        # first-step None sentinel (the l_1st_euler Euler-dt start) would be
        # a None->Field transition that crashes lax.scan.  With Nbb seeded = Nnn,
        # the scan's first step is a 2dt leapfrog from Nbb=Nnn (a stable forward
        # start; the eager run_dino loop keeps the exact NEMO Euler-dt start via
        # the None sentinel).  No-op when the leapfrog family is off or already
        # seeded.
        if (getattr(self.config, "outer_integrator", "forward_euler")
                in ("leapfrog", "nemo_mlf") and state.u_before is None):
            # Seed with the source Fields directly (name 'u'/'v'/…) so the carry
            # treedef EXACTLY matches what _leapfrog_step writes each step
            # (u_before=state.u.replace(data=…), i.e. name 'u') — a name mismatch
            # ('u_before' vs 'u') is custom-node aux data and would fail the scan
            # constant-pytree reconciliation.
            state = state._replace(
                u_before=state.u, v_before=state.v, T_before=state.T,
                S_before=state.S, eta_before=state.eta,
            )
            # KNOWN, NAMED, NOT FIXED HERE (#1729, review finding). Seeding
            # before:=now makes ``_euler_start`` False, so a scan-driven run
            # from rest takes a 2dt FILTERED leap-frog first step, where NEMO
            # takes a 1dt unfiltered Euler one (stpmlf.f90:131-133 plus the
            # three filter guards). The seed is required -- lax.scan needs a
            # constant carry treedef, and the None -> Field transition breaks
            # it -- so closing this needs a traced first-step flag, not a
            # different seed. It predates #1729 and is unchanged by it.
            # SCOPE: only a driver that calls this reaches it, i.e.
            # barotropic_solver="rigid_lid" or barotropic_slow_forcing_ab2
            # (run_dino.py). The NEMO-faithful DINO card is neither
            # (explicit_substep, slow_forcing_ab2 False) and runs the eager
            # loop, so its first step is the Euler start.

        # barotropic_forcing_centred (#1226 item 3): seed tau_x_prev/
        # tau_y_prev/freshwater_eta_prev from THIS step's forcing (NEMO
        # nit000 rule, sbcmod.F90:568-573 -- "before" set equal to "now" on
        # the very first call, no restart) so a scan driver's first
        # centred step degenerates to plain NOW exactly like the eager
        # ``_leapfrog_step`` Euler start. Reads
        # ``surface_forcing``/``freshwater`` out of ``step_kwargs`` (the
        # SAME forcing the scan will pass to ``step()``); a driver that
        # varies forcing per scan iteration (xs=...) rather than a fixed
        # kwarg must seed these fields itself before the scan starts.
        # No-op when the flag is off or already seeded (idempotent re-seed).
        if (getattr(self.config, "barotropic_forcing_centred", False)
                and (state.tau_x_prev is None
                     or state.freshwater_eta_prev is None)):
            _seed = _seed_centred_forcing_carry(
                step_kwargs.get("surface_forcing"),
                step_kwargs.get("freshwater"),
                self.config.rho_0, state.land_mask.data)
            if state.tau_x_prev is None and "tau_x_prev" in _seed:
                state = state._replace(tau_x_prev=_seed["tau_x_prev"],
                                       tau_y_prev=_seed["tau_y_prev"])
            if (state.freshwater_eta_prev is None
                    and "freshwater_eta_prev" in _seed):
                state = state._replace(
                    freshwater_eta_prev=_seed["freshwater_eta_prev"])

        # Barotropic slow-forcing AB2 carry (Gᵁ time-centering): seed the prev
        # F_slow Fields to ZERO so the scan carry pytree is stable from step 1
        # (the step stores a Field every step when barotropic_slow_forcing_ab2
        # is on; a None -> Field transition mid-scan crashes lax.scan, and
        # _step_impl raises on an unseeded prev).  First step then applies
        # (3/2+ε)·F_slow — the same AB2 cold-start convention as the outer
        # integrator.  Mirrors the in-driver seeding of
        # build_silvestri_baroclinic_jet_setup; seeding here makes the flag
        # usable by ANY seed_scan_carry driver (e.g. run_dino).  Field metadata
        # matches the step's own storage (dims ("lat","lon_u")/("lat_v","lon"),
        # units m/s^2).  No-op when the flag is off or already seeded.
        if (getattr(self.config.barotropic, "barotropic_slow_forcing_ab2", False)
                and (state.F_slow_u_prev is None or state.F_slow_v_prev is None)):
            # Seed as a PAIR: a partial carry (one Field, one None -- e.g. a
            # hand-built restart) would skip a u-only guard and then raise in
            # _step_impl / flip None->Field mid-scan (codex).  Preserve an
            # already-seeded component; zero-fill only the missing one.
            from legoesm.core.field import Field
            _fu = state.F_slow_u_prev
            _fv = state.F_slow_v_prev
            if _fu is None:
                _fu = Field(data=jnp.zeros_like(state.u.data[:, :, 0]),
                            name="F_slow_u_prev", dims=("lat", "lon_u"),
                            units="m/s^2")
            if _fv is None:
                _fv = Field(data=jnp.zeros_like(state.v.data[:, :, 0]),
                            name="F_slow_v_prev", dims=("lat_v", "lon"),
                            units="m/s^2")
            state = state._replace(F_slow_u_prev=_fu, F_slow_v_prev=_fv)

        # Prognostic-EKE carry: when EKE is on but the eddy-energy field has not
        # been seeded (state.eke is None), pre-seed it to the e_min floor so the
        # scan keeps a constant pytree (the model step would otherwise turn
        # eke None -> Field on the first iteration, which crashes lax.scan).  The
        # shape is STATIC per config: 3-D (n_lat, n_lon, nlev-1) at the interior
        # interfaces when eke_3d, else 2-D (n_lat, n_lon) — so the carried shape
        # is fixed for the whole scan.
        gm_redi = getattr(self.config, "gm_redi", None)
        if (gm_redi is not None and gm_redi.eke is not None
                and state.eke is None):
            from legoesm.core.field import Field
            lm = state.land_mask.data
            e_min = gm_redi.eke.e_min
            if gm_redi.eke.eke_3d:
                nlev = state.T.data.shape[-1]
                eke0 = jnp.where(
                    lm[:, :, jnp.newaxis] > 0.5, e_min, 0.0,
                ) * jnp.ones((1, 1, nlev - 1), dtype=lm.dtype)
                eke_dims = ("lat", "lon", "level")
            else:
                eke0 = e_min * lm
                eke_dims = ("lat", "lon")
            state = state._replace(
                eke=Field(data=eke0, name="eke", dims=eke_dims,
                          units="m^2/s^2"))
            # The 3-D EKE step also writes state.eke_diss (Veros eke_diss_iw) every
            # step; seed it to zero here so the None -> Field transition never
            # happens mid-scan (breaks the constant-pytree carry). Only the 3-D
            # path produces it, so seed it only there.
            if gm_redi.eke.eke_3d and state.eke_diss is None:
                ediss0 = jnp.zeros((lm.shape[0], lm.shape[1], nlev - 1),
                                   dtype=eke0.dtype)
                state = state._replace(
                    eke_diss=Field(data=ediss0, name="eke_diss",
                                   dims=("lat", "lon", "level"),
                                   units="m^2/s^3"))

        # Prognostic-TKE carry: when the prognostic TKE closure is on but the
        # TKE field has not been seeded (state.tke is None), pre-seed it (and the
        # eke_diss source-carry field, if source_eke_diss) so the scan keeps a
        # constant pytree (the model step would otherwise turn tke None -> Field
        # on the first iteration, crashing lax.scan). Shapes are STATIC per
        # config: 3-D (n_lat, n_lon, nlev-1) at the interior interfaces. TKE
        # seeded at tke_background on wet columns; eke_diss seeded at 0.
        if self._tke_prognostic_active() and state.tke is None:
            from legoesm.core.field import Field
            lm = state.land_mask.data
            nlev = state.T.data.shape[-1]
            dtype = state.T.data.dtype
            tke_cfg = self.config.physics.vertical_mixing.tke
            wet3 = (lm[:, :, jnp.newaxis] > 0.5)
            tke0 = jnp.where(
                wet3, tke_cfg.tke_background, 0.0,
            ).astype(dtype) * jnp.ones((1, 1, nlev - 1), dtype=dtype)
            state = state._replace(
                tke=Field(data=tke0, name="tke",
                          dims=("lat", "lon", "level"), units="m^2/s^2"))
            if (getattr(tke_cfg, "source_eke_diss", False)
                    and state.eke_diss is None):
                ediss0 = jnp.zeros((lm.shape[0], lm.shape[1], nlev - 1),
                                   dtype=dtype)
                state = state._replace(
                    eke_diss=Field(data=ediss0, name="eke_diss",
                                   dims=("lat", "lon", "level"),
                    units="m^2/s^3"))

        # Same cold-start coefficient memory as direct step(); idempotent for
        # restart bridges and already-seeded scan carries.
        state = self._seed_tke_preclosure_carry(state)

        # TKE-advection AB2 carry: seed the prior advective tendency dtke to
        # zero (Veros's zero-initialised dtke[taum1]) so the scan pytree stays
        # constant (the step writes a dtke Field every iteration when the
        # advection is active). Separate from the tke seeding above: state.tke
        # may have been pre-seeded by the caller while dtke was not.
        if self._tke_advection_active() and state.dtke is None:
            from legoesm.core.field import Field
            lm = state.land_mask.data
            nlev = state.T.data.shape[-1]
            dtype = state.T.data.dtype
            dtke0 = jnp.zeros((lm.shape[0], lm.shape[1], nlev - 1),
                              dtype=dtype)
            state = state._replace(
                dtke=Field(data=dtke0, name="dtke",
                           dims=("lat", "lon", "level"), units="m^2/s^3"))

        # store_mass_flux carry (#1442): the step turns mass_flux_u/v from
        # None into Fields, so an unseeded carry is a None -> Field transition
        # that crashes lax.scan AND retraces a direct jitted step loop.  Seed
        # to zeros through the SAME constructor the step writes with, so the
        # treedef (Field name/dims/units are aux data) matches exactly.  The
        # seed value is unreachable: nothing READS these slots.  The salt
        # sibling (store_salt_flux) seeds through ITS shared constructor for
        # the identical reason.
        state = seed_mass_flux_carry(
            state, getattr(self.config, "store_mass_flux", False))
        state = seed_salt_flux_carry(
            state, getattr(self.config, "store_salt_flux", False))

        # Rigid-lid: pre-build the static island/depth data (host-side
        # flood-fill) so the scan captures it as a compile-time constant, and
        # seed the streamfunction carry (ψ, dψ, dψ_prev, dpsin, dpsin_prev) to
        # zero so the carry pytree is constant across iterations (None -> array
        # would crash lax.scan).  Cold-start ψ=0 (rest); the AB2 history is 0.
        if self.config.barotropic.barotropic_solver == "rigid_lid":
            rl = self._ensure_rigid_lid_data(state)
            if state.psi is None:
                _dt_rl = state.u.data.dtype
                _zV = jnp.zeros((self.grid.n_lat + 1, self.grid.n_lon + 1),
                                dtype=_dt_rl)
                _zI = jnp.zeros((rl.nisle,), dtype=_dt_rl)
                state = state._replace(
                    psi=_zV, dpsi=_zV, dpsi_prev=_zV, dpsin=_zI, dpsin_prev=_zI)

        # --- dtype reconciliation: carry-in dtype == carry-out dtype ----------
        # The structural seeding above fixes the TREEDEF; this fixes the DTYPES.
        # 1) Lift every float leaf to the working precision (the top of the
        #    state⊕geometry dtype lattice — f64 under x64 because the z-coordinate
        #    is f64; f32 on an all-f32 build, making the rest a no-op).
        work_dtype = jnp.result_type(state.T.data.dtype, self.z_coord.dz_ref.dtype)
        state = jax.tree_util.tree_map(
            lambda a: a.astype(work_dtype)
            if jnp.issubdtype(a.dtype, jnp.floating) else a,
            state,
        )
        # 2) Ask the step (abstractly, no FLOPs) what dtype each leaf becomes —
        #    the dynamics stay at work_dtype, EKE / eke_diss / streamfunction get
        #    pinned back to storage — and cast the seed to match. Starting from
        #    the lattice top makes this a single self-consistent pass.
        target = jax.eval_shape(lambda s: self.step(s, dt, **step_kwargs), state)
        state = jax.tree_util.tree_map(
            lambda a, t: a.astype(t.dtype), state, target,
        )
        return state

    def integrate_scan(
        self,
        state: LatLonCGridOceanState,
        n_steps: int,
        dt: float,
        t0_seconds: float = 0.0,
    ) -> tuple[LatLonCGridOceanState, LatLonCGridOceanState]:
        """Integrate using jax.lax.scan (differentiable).

        Parameters
        ----------
        state : LatLonCGridOceanState
        n_steps : int
        dt : float

        Returns
        -------
        final_state, trajectory (stacked)

        Notes
        -----
        For AB2: the initial carry must have Field (not None) for
        ``T_flux_div_prev`` / ``S_flux_div_prev`` so the pytree
        structure is stable across scan iterations.  If they are None,
        this method pre-initializes them with zero-filled Fields.
        The first step then uses AB2 with zero previous tendency,
        giving an effective coefficient of (3/2+eps) ≈ 1.6 rather
        than the Euler fallback of 1.0.  At typical CFL values
        (≤ 0.3) this is stable.
        """
        _tf = getattr(self.config, "tidal_forcing", None)
        _tide_on = _tf is not None and _tf.enabled
        # seed_scan_carry probes the step via jax.eval_shape — with the tide
        # on it must probe the SAME (t-threaded) signature, both so the shape
        # discovery follows the tide-on trace and so the step()'s eager
        # enabled-but-no-time guard does not trip inside the probe.
        state = (self.seed_scan_carry(state, dt,
                                      t_seconds=jnp.asarray(t0_seconds))
                 if _tide_on else self.seed_scan_carry(state, dt))

        if _tide_on:
            # Feed the traced per-step elapsed model time through xs (NOT a scan
            # carry change) so the equilibrium tide advances each step; grads
            # flow back to t and the config amplitudes. t=0 <-> the constituents'
            # reference epoch. Tide-off keeps xs=None => bit-identical.
            times = t0_seconds + dt * jnp.arange(n_steps)

            def scan_fn(state, t):
                new_state = self.step(state, dt, t_seconds=t)
                return new_state, new_state

            final_state, trajectory = jax.lax.scan(scan_fn, state, xs=times)
            return final_state, trajectory

        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, xs=None, length=n_steps,
        )
        return final_state, trajectory


class _NEMOWSTracerSolveTrace(NamedTuple):
    """Consumed production-JIT tracer ZDF operands for Round 126."""

    heat_K: object
    isoneutral_K: object
    effective_K: object
    e3t_after: object
    e3w_now: object
    wet: object
    content_T: object
    lower: object
    diagonal: object
    upper: object
    solved_T: object
    viscosity_K: object


class _NEMOWSQsrAssociationTrace(NamedTuple):
    """Consumed production-JIT QSR source-association arrays for Round 190."""

    tendency_kbb: object
    qsr_kbb: object
    qsr_kmm: object
    thickness_kbb: object
    thickness_kmm: object
    process_qsr_kbb: object
    process_surface_rate: object
    process_qsr_rate: object


class _NEMOWSTracerProcessTrace(NamedTuple):
    """Private production-JIT stage-3 temperature boundaries for Round 124."""

    state_after: object
    Tbb: object
    q_Kbb: object  # noqa: N815 - NEMO time-level spelling is the record API.
    q_Kmm: object  # noqa: N815 - NEMO time-level spelling is the record API.
    q_Kaa: object  # noqa: N815 - NEMO time-level spelling is the record API.
    boundaries: object
    Taa: object
    qsr_association: object
    vertical_solve: object
    fct_activity: object
    ldf_diagnostics: object


class _NEMOWSLdfDiagnosticTrace(NamedTuple):
    """Write-only production-JIT LDF internals without the QSR observer."""

    state_after: object
    ldf_diagnostics: object
