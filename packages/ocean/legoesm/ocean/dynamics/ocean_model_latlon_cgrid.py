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
    diagnose_w_from_flux_div,
    flux_form_vertical_tracer_advection,
    flux_form_vertical_tracer_advection_tvd,
    flux_form_vertical_tracer_advection_centered,
)
from legoesm.ocean.state import (
    LatLonCGridOceanState,
    LatLonCGridOceanConfig,
    constants_equal,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    latlon_cgrid_ocean_baroclinic_tendencies,
    compute_frozen_geom_density,
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
    min_cell_to_uface,
    min_cell_to_vface,
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
from legoesm.ocean.freshwater import freshwater_eta_tendency, virtual_salt_flux
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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute advection flux divergence for a single tracer field.

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

    if tracer_advection in ("ppm_fct", "fct2"):
        from legoesm.ocean.advection import fct_tracer_advection
        div_hut, vert_flux_div = fct_tracer_advection(
            tr, mass_flux_u, mass_flux_v, w_baro, h_k_old, grid, dt,
            high_order="ppm" if tracer_advection == "ppm_fct" else "centred2",
            tracer_before=tr_before,
            # #1226 item 8: same wet mask already threaded as
            # recon_fill_mask (is_active/active_3d) — faithfully masks
            # the NEMO nonosc per-point bound at dry cells (see
            # fct_tracer_advection's active_mask docstring).
            active_mask=recon_fill_mask,
        )
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

    if linssh_top_flux:
        # NEMO key_linssh top-cell concentration/dilution: the surface
        # vertical advective flux is F[0] = w0*T0 (first-order, OUTSIDE any
        # limiter — traadv_fct.F90:413-423 with a zero antidiffusive top
        # flux), instead of the rigid F[0]=0 of the stretching-column z*.
        # vert_flux_div[k] = F[k] - F[k+1], so add w0*T0 to level 0.
        # w_baro[...,0] = deta/dt (fixed-thickness continuity); zero on land.
        vert_flux_div = vert_flux_div.at[..., 0].add(
            w_baro[..., 0] * tr[..., 0])

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
):
    """Advection flux divergence for TWO tracers (T, S) in one pass.

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
        tracer_advection not in _LEVEL_SEPARABLE_H_SCHEMES
        or os.environ.get("LEGOESM_TRACER_PAIR", "0") != "1"
    ):
        return (
            _compute_advection_flux_div(
                tr_a, tracer_advection, mass_flux_u, mass_flux_v,
                w_baro, h_k_old, h_u_old, h_v_old, grid, dt,
                recon_fill_mask=recon_fill_mask,
                linssh_top_flux=linssh_top_flux,
                tr_before=tr_a_before,
            ),
            _compute_advection_flux_div(
                tr_b, tracer_advection, mass_flux_u, mass_flux_v,
                w_baro, h_k_old, h_u_old, h_v_old, grid, dt,
                recon_fill_mask=recon_fill_mask,
                linssh_top_flux=linssh_top_flux,
                tr_before=tr_b_before,
            ),
        )

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
    content = (h_before * t_before
               + (h_after * t_expl - h_now * t_now)
               + h_now * d_diss)
    return jnp.where(mask > 0, content / jnp.maximum(h_after, h_floor), t_now)


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
    ):
        self.z_coord = z_coord
        self.config = config or LatLonCGridOceanConfig.from_flat()
        self._validate_config(self.config)
        # Convert LatLonGrid -> LatLonCGridGeometry once at construction.
        # All downstream operators see the enriched geometry with per-cell
        # metric arrays.  For a plain LatLonGrid this is a no-op on field
        # access (legacy fields are identical); for a tripolar grid the
        # geometry carries fold descriptor and rotation angles.
        # ``metric_convention`` (#1226) is validated above, so the raise on
        # an unknown value happens before this call.
        self.grid = ensure_geometry(
            grid, metric_convention=self.config.metric_convention)
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
    def _validate_config(config: LatLonCGridOceanConfig) -> None:
        """Validate configuration ranges."""
        nonnegative = {
            "A_h": config.lateral_viscosity.A_h,
            "B_h": config.lateral_viscosity.B_h,
            "K_h": config.K_h,
            "A_v": config.A_v,
            "K_v": config.K_v,
            "hyperdiff_coeff": config.hyperdiff_coeff,
            "barotropic_diffusion_alpha": config.barotropic.barotropic_diffusion_alpha,
        }
        for name, value in nonnegative.items():
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

        # #1226: T/u-face metric convention dispatch -- raise on an unknown
        # value rather than silently falling through to create_latlon_geometry's
        # own raise deep inside ensure_geometry (fail at config-validation time,
        # before any grid conversion work happens).
        if config.metric_convention not in ("exact", "nemo_isotropic"):
            raise ValueError(
                "metric_convention must be 'exact' or 'nemo_isotropic', got "
                f"{config.metric_convention!r}"
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
        _valid_fw = {"none", "virtual_salt_flux"}
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
        if (getattr(config, "freshwater_salinity", "s_ref") == "local"
                and bool(getattr(config, "normalize_freshwater", False))):
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
        _vert_mom_scheme = getattr(
            config, "vertical_momentum_scheme", "upwind_perturbation")
        if _vert_mom_scheme not in VALID_VERTICAL_MOMENTUM_SCHEME:
            raise ValueError(
                f"vertical_momentum_scheme must be one of "
                f"{sorted(VALID_VERTICAL_MOMENTUM_SCHEME)}, "
                f"got {_vert_mom_scheme!r}",
            )
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
                               "nemo_boxcar_ab3"}
        if (getattr(config, "surface_stress_implicit", False)
                and not getattr(config.barotropic,
                                "nemo_stage_mean_imposition", False)
                and config.barotropic.barotropic_solver != "explicit_substep"):
            # Under the split-explicit free surface ("explicit_substep") the
            # post-solve depth-mean shift is NEMO's OWN MLF arrangement
            # (stpmlf.F90: dyn_spg THEN dyn_zdf, no post-zdf re-imposition —
            # the shift feeds the next step's Kbb barotropic seed), and
            # F_slow carries the wind for the substeps exactly as NEMO's
            # zu_frc wind term (dynspg_ts.F90 ~L360).  So no stage-mean
            # imposition is required there; other solvers keep the guard.
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
                in ("nemo_ab3am4", "nemo_boxcar_ab3")
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
            if _bd_scheme not in ("nemo_quadratic", "nemo_loglayer"):
                raise ValueError(
                    "zdf_drag_in_matrix=True requires bottom_drag_scheme in "
                    '{"nemo_quadratic", "nemo_loglayer"} (NEMO\'s zdfdrg '
                    f"rCdU_bot rate); got {_bd_scheme!r}.")
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
        _leapfrog_family = ("leapfrog", "nemo_mlf")
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
            if not getattr(config, "implicit_vmix_e3t_now_divisor", False):
                raise ValueError(
                    'outer_integrator="nemo_mlf" requires '
                    "implicit_vmix_e3t_now_divisor=True: nemo_mlf is a literal "
                    "transcription of stpmlf.F90's dyn_zdf/tra_zdf calls, whose "
                    "implicit-solve gradient divisor is e3w(Kmm) (trazdf.F90:"
                    "219-220), not legoESM's default after-solve midpoint slot. "
                    "Set implicit_vmix_e3t_now_divisor=True.")
            # mlf_baro_corr (stpmlf.F90 row 30) is WAIVED inside _nemo_mlf_step
            # citing W1a -- PROVABLY a no-op only because no depth-mean source
            # exists under surface_stress_implicit=False (see the method's own
            # docstring). A future card flipping surface_stress_implicit=True
            # together with nemo_mlf would silently make that waiver WRONG (a
            # real depth-mean source would then need the corrector call this
            # method does not build) -- reject rather than let the waiver rot.
            if getattr(config, "surface_stress_implicit", False):
                raise ValueError(
                    'outer_integrator="nemo_mlf" does not support '
                    "surface_stress_implicit=True: _nemo_mlf_step WAIVES the "
                    "mlf_baro_corr call (stpmlf.F90 row 30) citing W1a -- "
                    "provably a no-op ONLY when surface_stress_implicit=False "
                    "(no depth-mean source exists in lego's solve to "
                    "reconcile). Enabling surface_stress_implicit would create "
                    "a real depth-mean source with no corrector to remove it. "
                    "Build the mlf_baro_corr kernel (spec §2/§6-5) before "
                    "lifting this guard.")
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
                    == "nemo_before" and _outer_int not in _leapfrog_family):
                raise ValueError(
                    'vertical_mixing.tke.tke_n2_time_level="nemo_before" '
                    'requires outer_integrator in ("leapfrog", "nemo_mlf"): '
                    "the true rn2b (Nbb) tracers only exist as "
                    "state.T_before/S_before under the leap-frog-family "
                    f"(NEMO Modified-Leap-Frog) time integrator. Got "
                    f"outer_integrator={_outer_int!r}.")
            _tke_shear_ctor = getattr(_tke_cfg_ctor, "tke_shear_production",
                                     "squared_centered")
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
                # so they cancel at the pre-step state. Under vorticity_scheme=
                # "{ene,een}_total" the planetary term is the vertex-f TRANSPORT-
                # form EEN flux, so the subtraction must ALSO be EEN
                # (barotropic.barotropic_coriolis="een"); the legacy 4-pt-avg
                # face-f "avg" stencil would leave an O(1) residual Coriolis.
                if getattr(config.barotropic, "barotropic_coriolis",
                           "avg") not in ("een", "een_metric"):
                    raise ValueError(
                        f'vorticity_scheme="{_vs}" with '
                        'barotropic_coriolis_split="live" requires '
                        'barotropic.barotropic_coriolis="een"/"een_metric" (NEMO '
                        "dyn_spg_ts::dyn_cor_2D EEN): the _total planetary "
                        "term is the vertex-f EEN transport-form flux, so the "
                        "live pre-step subtraction must use the SAME EEN "
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
            if getattr(config.barotropic, "barotropic_time_filter",
                       "cosine") == "nemo_ab3am4":
                raise ValueError(
                    'barotropic_coriolis_split="live" is incompatible with '
                    'barotropic_time_filter="nemo_ab3am4": the AB3 substep '
                    "applies the live Coriolis to the EXTRAPOLATED mid-step "
                    "velocity U_mid while the pre-step subtraction uses the "
                    "plain pre-step U_bar, so substep-0 would not cancel "
                    "bit-exactly. Use the boxcar filter (nn_bt_flt=2, NEMO's "
                    "DINO selection) with the live split.")
        _valid_time_int = {"euler", "ab2", "rk3"}
        if config.tracer_time_integrator not in _valid_time_int:
            raise ValueError(
                f"tracer_time_integrator must be one of {_valid_time_int}, "
                f"got {config.tracer_time_integrator!r}")
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

        # NEMO-faithful e3w(Kmm) divisor (#1226 W1): same "needs an implicit
        # solve to host it" reasoning as implicit_vmix_dzw_slot above, plus
        # mutual exclusion — the two flags pick DIFFERENT divisor slots
        # (Veros dzw vs NEMO e3w(Kmm)); selecting both is ambiguous, not a
        # silent priority order.
        if (getattr(config, "implicit_vmix_e3t_now_divisor", False)
                and not config.implicit_vertical_mixing):
            raise ValueError(
                "implicit_vmix_e3t_now_divisor=True requires "
                "implicit_vertical_mixing=True: the NEMO e3w(Kmm) gradient "
                "slot is the divisor of the backward-Euler tracer/momentum-"
                "friction vertical-diffusion solve (trazdf.F90:219-220). With "
                "explicit vertical mixing there is no implicit solve to host "
                "it. Set implicit_vertical_mixing=True, or "
                "implicit_vmix_e3t_now_divisor=False to keep the midpoint "
                "slot.")
        if (getattr(config, "implicit_vmix_dzw_slot", False)
                and getattr(config, "implicit_vmix_e3t_now_divisor", False)):
            raise ValueError(
                "implicit_vmix_dzw_slot and implicit_vmix_e3t_now_divisor are "
                "mutually exclusive: both select the implicit-solve gradient "
                "divisor (Veros dzw vs NEMO e3w(Kmm)) — set at most one.")

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
                   ldf_state=None):
        """Compute baroclinic tendencies.

        ``momentum_only=True`` skips the (T/S-frozen) tracer-diffusion
        tendency for the RK3 momentum sub-stages, which discard dT_dt/dS_dt
        — bit-identical du_dt/dv_dt, fewer halos/compute (see
        ``latlon_cgrid_ocean_baroclinic_tendencies``).

        ``precomputed_geom_density`` (a ``(J, h_k, rho_prime,
        p_prime_filled)`` tuple from :func:`compute_frozen_geom_density`)
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
        _grid = grid if grid is not None else self.grid
        _vmask = vertex_mask if vertex_mask is not None else self._vertex_mask
        # ``ab2_scope_override`` (leap-frog Nbb-diffusion pass): the du_dt/du_diss
        # split is keyed off ``config.ab2_scope`` inside the tendency kernel. To
        # run the split WITHOUT mutating the model's config, overlay just that one
        # field on a copy (a NamedTuple ``_replace`` — every other field
        # identical). ``None`` ⇒ pass the model config unchanged ⇒ bit-identical.
        _cfg = (self.config._replace(ab2_scope=ab2_scope_override)
                if ab2_scope_override is not None else self.config)
        return latlon_cgrid_ocean_baroclinic_tendencies(
            state, _grid, self.z_coord, _cfg,
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
        )

    def tendencies_with_diagnostics(
        self, state: LatLonCGridOceanState, surface_forcing=None,
        sponge=None, dt=300.0, *, grid=None, vertex_mask=None,
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

        ``grid``/``vertex_mask`` (optional, SPMD): default ``None`` →
        ``self.grid``/``self._vertex_mask`` (bit-identical); a band-local
        grid is injected by a future ``shard_map`` wrapper.
        """
        _grid = grid if grid is not None else self.grid
        _vmask = vertex_mask if vertex_mask is not None else self._vertex_mask
        return latlon_cgrid_ocean_baroclinic_tendencies(
            state, _grid, self.z_coord, self.config,
            physics_fn=self._physics_fn,
            surface_forcing=surface_forcing,
            sponge=sponge,
            dt=dt,
            diagnose_momentum=True,
            surface_tracer_forcing_fn=self._surface_tracer_forcing_fn,
            vertex_mask=_vmask,
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
                   _ldf_state=None):
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
                  else getattr(self.config, "ab2_scope", "total"))
        # SPMD resolve: default None → the model's own grid / vertex mask
        # (bit-identical single-device path).
        _grid = grid if grid is not None else self.grid
        _vmask = vertex_mask if vertex_mask is not None else self._vertex_mask
        # Prescribed-flow lever (config.prescribed_flow, validated at
        # construction).  STATIC Python gate on the config value (CLAUDE.md
        # feature-gating exception): None (default) leaves every gated block
        # below untraced — the compiled graph is byte-identical to before the
        # lever existed.  Never jnp.where/lax.cond here.
        _pflow = getattr(self.config, "prescribed_flow", None)

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
        if (getattr(self.config, "barotropic_forcing_centred", False)
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
        dt_mom = dt / self.config.dt_mom_ratio

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
            state, _grid, self.z_coord, self.config)
        tend = self.tendencies(state, surface_forcing, sponge=sponge, dt=dt,
                               precomputed_geom_density=_geom_density,
                               grid=_grid, vertex_mask=_vmask,
                               ab2_scope_override=_ab2_scope_override,
                               ldf_state=_ldf_state)
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
            state.eta.data, state.H_bathy.data, self.z_coord,
            min_water_column_m=self.config.min_water_column_m,
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

        if getattr(self.config, "surface_stress_implicit", False):
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
                state.eta.data, state.H_bathy.data, self.z_coord)
            _sfx = (surface_stress_faces(
                        surface_forcing, du_dt.dtype, self.z_coord, _J_fs,
                        _grid)
                    if surface_forcing is not None else None)
            if _sfx is not None:
                _tau_i_u, _tau_j_v, _, _ = _sfx
                _r0 = jnp.asarray(self.config.constants.rho_0,
                                  dtype=du_dt.dtype)
                F_slow_u = F_slow_u + _tau_i_u / (_r0 * H_u_pre) \
                    * state.u_mask.data
                F_slow_v = F_slow_v + _tau_j_v / (_r0 * H_v_pre) \
                    * state.v_mask.data

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
        if getattr(self.config, "barotropic_drag_substep", False):
            from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
                nemo_bottom_drag_rate_faces,
            )
            _centred_drag = (
                getattr(self.config, "barotropic_forcing_centred", False)
                and getattr(state, "u_before", None) is not None
                and getattr(state, "v_before", None) is not None)
            _u_src = state.u_before.data if _centred_drag else state.u.data
            _v_src = state.v_before.data if _centred_drag else state.v.data
            _r_u_bt, _r_v_bt, _isb_u, _isb_v = nemo_bottom_drag_rate_faces(
                state.u.data, state.v.data, h_k_pre, self.z_coord,
                self.config, _grid)
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

        # A2 — depth-mean biharmonic hyperviscosity on (U_bar, V_bar).
        # Damps the barotropic standing mode at deep cells next to steep
        # slopes (Rhines 1969 bottom-trapped wave with f≈0) without
        # touching the baroclinic perturbation u' (already finalized
        # above as du_dt_pert / dv_dt_pert).  Applied as an additional
        # slow forcing on the implicit-CN barotropic solver:
        #   ∂U_bar/∂t |_diss = -ν₄ · ∇⁴ U_bar.
        # MOM6/HIM BIHARMONIC_BAROTROPIC analog.  No-op at default
        # ``B_h_barotropic = 0`` (bit-exact backward compat).
        if getattr(self.config.lateral_viscosity, "B_h_barotropic", 0.0) > 0.0:
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
                self.config.lateral_viscosity.B_h_barotropic, dtype=F_slow_u.dtype,
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
        if getattr(self.config.barotropic, "barotropic_slow_forcing_ab2", False):
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
            _eps = self.config.ab2_epsilon
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
        if getattr(self.config, "momentum_time_integrator", "euler") == "rk3":
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
                                     grid=_grid, vertex_mask=_vmask)
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
        elif getattr(self.config, "momentum_time_integrator",
                     "euler") == "rk3_ws":
            # NEMO stprk3_stg Wicker-Skamarock RK3: every stage restarts from
            # u0 with the PREVIOUS stage's RHS and the stage dt (dt/3, dt/2,
            # dt) — NOT Shu-Osher convex combinations. Per-stage RHS content
            # mirrors NEMO exactly: stage 1 = the precomputed full tendency
            # (stp2d's Ue_rhs INCLUDES dyn_ldf); stage 2 = hpg+vor+adv ONLY
            # (NO lateral viscosity, stprk3_stg:318-334); stage 3 = full again
            # (dyn_ldf re-applied). ZDF (the implicit vertical solve) runs once
            # after the momentum stages == NEMO's stage-3-only dyn_zdf. The
            # linear stability polynomial R(z)=1+z+z^2/2+z^3/6 is identical to
            # SSP-RK3, so the explicit_ab2 Coriolis coupling bound (f*dt<=
            # sqrt(3)) carries over.
            u0 = state.u.data
            v0 = state.v.data

            def _mom_pert_ws(u_in, v_in, skip_ldf):
                st = state._replace(
                    u=state.u.replace(data=u_in * u_mask_3d),
                    v=state.v.replace(data=v_in * v_mask_3d),
                )
                td = self.tendencies(st, surface_forcing, sponge=sponge, dt=dt,
                                     momentum_only=True,
                                     precomputed_geom_density=_geom_density,
                                     grid=_grid, vertex_mask=_vmask,
                                     skip_lateral_viscosity=skip_ldf)
                _du = td.du_dt.data
                _dv = td.dv_dt.data
                _Fu = jnp.sum(_du * h_u_pre, axis=-1) / H_u_pre * state.u_mask.data
                _Fv = jnp.sum(_dv * h_v_pre, axis=-1) / H_v_pre * state.v_mask.data
                return _du - _Fu[..., jnp.newaxis], _dv - _Fv[..., jnp.newaxis]

            # stage 1 (dt/3), RHS = the stage-1 full tendency (incl. LDF)
            u1 = u0 + (dt_mom / 3.0) * du_dt_pert
            v1 = v0 + (dt_mom / 3.0) * dv_dt_pert
            # stage 2 (dt/2), RHS(u1) WITHOUT lateral viscosity
            p1u, p1v = _mom_pert_ws(u1, v1, True)
            u2 = u0 + (dt_mom / 2.0) * p1u
            v2 = v0 + (dt_mom / 2.0) * p1v
            # stage 3 (dt), RHS(u2) with lateral viscosity
            p2u, p2v = _mom_pert_ws(u2, v2, False)
            u_star = u0 + dt_mom * p2u
            v_star = v0 + dt_mom * p2v
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
        if getattr(self.config, "coriolis_scheme", "matsuno_split") != "explicit_ab2":
            u_star, v_star = _forward_backward_coriolis_3d(
                u_star, v_star, dt_mom, _grid, self.z_coord, self.config,
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
            state_mid.eta.data, state_mid.H_bathy.data, self.z_coord,
            min_water_column_m=self.config.min_water_column_m,
        )

        # 6. Barotropic step.  Two paths:
        #    - explicit_substep: split-explicit forward-backward substepping
        #      with cosine/box time filter.
        #    - implicit_cn: single-step Crank-Nicolson free surface (PCG).
        #      Eliminates the chequerboard mode by construction; no
        #      substepping or time filter needed.

        # Freshwater mass flux for barotropic continuity equation
        F_slow_eta = None
        if freshwater is not None and self.config.freshwater_closure != "none":
            F_slow_eta = freshwater_eta_tendency(
                freshwater, self.config.rho_0,
            ) * state.land_mask.data
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
            if (getattr(self.config, "barotropic_forcing_centred", False)
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

        if self.config.barotropic.barotropic_solver == "rigid_lid":
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
                getattr(self.config, "coriolis_scheme", "matsuno_split")
                != "explicit_ab2")
            state_new, (Hu_avg, Hv_avg) = barotropic_rigid_lid_latlon_cgrid(
                state_mid, dt_mom, _grid, self.z_coord, self.config, rl_data,
                F_slow_u=F_slow_u, F_slow_v=F_slow_v,
                add_barotropic_coriolis=_add_bt_cor,
            )
        elif self.config.barotropic.barotropic_solver == "implicit_cn":
            state_new, (Hu_avg, Hv_avg) = barotropic_implicit_latlon_cgrid(
                state_mid, dt_mom,
                _grid, self.z_coord, self.config,
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
            _nbaro = (self.config.barotropic.n_barotropic_substeps
                      * _barotropic_substep_scale)
            dt_s = dt_mom / _nbaro
            # Under coriolis_scheme="explicit_ab2" the planetary Coriolis already
            # reaches the barotropic mode via F_slow (its depth-mean came through
            # du_dt), so the substep must NOT add its own f×U_bt — this is the
            # Oceananigans split-explicit convention and removes the C-grid
            # 4-point Coriolis rotational null mode (the 2Δx barotropic mode that
            # otherwise blows the eddy-resolving jet).
            _add_bt_cor = (
                getattr(self.config, "coriolis_scheme", "matsuno_split")
                != "explicit_ab2")
            if (not _add_bt_cor) and getattr(
                    self.config, "barotropic_coriolis_split",
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
                    self.config.barotropic, "barotropic_coriolis", "avg")
                if _bt_cor_split in ("een", "een_metric"):
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
                        self.config.min_water_column_m, dtype=F_slow_u.dtype)
                    _cor_u_sub, _cor_v_sub = barotropic_coriolis_een_pre_step(
                        state_mid.u.data, state_mid.v.data, h_k_pre, _grid,
                        state.land_mask.data, state.u_mask.data,
                        state.v_mask.data, _min_wc, F_slow_u.dtype,
                        metric_complete=(_bt_cor_split == "een_metric"),
                        # Same EEN q-boundary / e3f rules as the 3-D EEN and
                        # as the substep loop's own _build_een_barotropic_inputs
                        # — otherwise this subtraction uses a DIFFERENT operator
                        # than the live term it is meant to cancel.
                        een_q_boundary=getattr(
                            self.config, "een_q_boundary", "neumann_fill"),
                        een_e3f_scheme=getattr(
                            self.config, "een_e3f_scheme", "min"),
                        # ...and the SAME dz_ref, or the fully-dry-vertex e3f
                        # differs from the live substep term this cancels.
                        dz_ref=getattr(self.z_coord, "dz_ref", None))
                    F_slow_u = (F_slow_u - _cor_u_sub) * state.u_mask.data
                    F_slow_v = (F_slow_v - _cor_v_sub) * state.v_mask.data
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
            if self.config.barotropic.barotropic_wide_halo:
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
                _baro_seed = dict(
                    _baro_seed, substep_scale=_barotropic_substep_scale)
            state_new, (Hu_avg, Hv_avg) = _baro_fn(
                state_mid, dt_s, _nbaro,
                _grid, self.z_coord, self.config,
                F_slow_eta=F_slow_eta,
                F_slow_u=F_slow_u,
                F_slow_v=F_slow_v,
                add_barotropic_coriolis=_add_bt_cor,
                t_seconds=t_seconds,  # traced model time for the equilibrium tide
                **_baro_seed,
            )

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
        if self.config.fix_eta_drift:
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
            if self.config.min_water_column_m is not None:
                from legoesm.ocean.dynamics.eta_floor import (
                    clamp_and_redistribute as _clamp_redistribute,
                )
                eta_floor = (
                    jnp.asarray(
                        self.config.min_water_column_m,
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
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            divergence_cgrid, interp_cell_to_uface,
        )

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
        if isinstance(self.z_coord, OceanPartialCellCoordinate):
            u_mask_3d_tracer, v_mask_3d_tracer = compute_face_masks_3d(
                self.z_coord.is_active, _grid,
            )
            u_mask_3d_tracer = u_mask_3d_tracer.astype(h_u_old.dtype)
            v_mask_3d_tracer = v_mask_3d_tracer.astype(h_v_old.dtype)
            active_3d = self.z_coord.is_active.astype(h_u_old.dtype)
        else:
            u_mask_3d_tracer = state.u_mask.data[..., jnp.newaxis]
            v_mask_3d_tracer = state.v_mask.data[..., jnp.newaxis]
            active_3d = mask_3d

        # Full 3D velocity (barotropic + baroclinic) from state after
        # barotropic correction.  The barotropic solver preserves the
        # baroclinic perturbation u' = u - U_bar and replaces the
        # barotropic component with the time-averaged U_bar_avg.
        u_3d = state_new.u.data   # (n_lat, n_lon+1, nlev)
        v_3d = state_new.v.data   # (n_lat+1, n_lon, nlev)

        # Correct the barotropic component so that depth-integrated
        # transport matches Hu_avg exactly.  The correction is the
        # difference between <H*U> (time-averaged transport) and
        # <U>*H (time-averaged velocity times pre-barotropic H).
        # H + Hu reductions per face share the h_u_old/h_v_old weight
        # on the level axis — fuse into one stack each.
        _u_pair = jnp.sum(jnp.stack([h_u_old, u_3d * h_u_old], axis=-1), axis=-2)
        H_u_old, Hu_3d = _u_pair[..., 0], _u_pair[..., 1]  # (n_lat, n_lon+1)
        _v_pair = jnp.sum(jnp.stack([h_v_old, v_3d * h_v_old], axis=-1), axis=-2)
        H_v_old, Hv_3d = _v_pair[..., 0], _v_pair[..., 1]  # (n_lat+1, n_lon)
        if _pflow is None:
            delta_U = (Hu_avg - Hu_3d) / jnp.maximum(H_u_old, 1e-10)
            delta_V = (Hv_avg - Hv_3d) / jnp.maximum(H_v_old, 1e-10)
            u_corrected = u_3d + delta_U[..., jnp.newaxis]
            v_corrected = v_3d + delta_V[..., jnp.newaxis]
        else:
            # Prescribed flow: NO barotropic transport correction.  u_3d/v_3d
            # already carry the pinned flow (splice above); Hu_avg/Hv_avg from
            # the DISCARDED barotropic solve must not leak into the tracer
            # mass fluxes.  mass_flux = h·u_pin exactly — for "zero" the
            # fluxes vanish and the diagnosed w below is identically zero.
            u_corrected = u_3d
            v_corrected = v_3d

        # Per-layer mass fluxes with full 3D velocity structure.
        # Unlike the previous barotropic-only distribution (which gave
        # uniform velocity at all depths and identically zero w),
        # this preserves baroclinic shear and produces non-zero vertical
        # velocity from Ekman pumping/suction.
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
            state_new.eta.data, state_new.H_bathy.data, self.z_coord,
            min_water_column_m=self.config.min_water_column_m,
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
        w_baro = diagnose_w_from_flux_div(
            flux_div_k, self.z_coord, thickness_weighted=True,
        )

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
        if (getattr(self.config, "adaptive_implicit_vertadv", False)
                and _pflow is None):
            from legoesm.ocean.dynamics.latlon_cgrid_operators import (
                interp_cell_to_vface,
            )
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
        # own small Euler correction).  GM bolus transport / K_33 / EKE
        # production (all row 27, Nnn) are UNCHANGED -- only the isoneutral
        # tendency's (T, S) READ swaps; ``density_jacobian``/``eta``/``kappa``
        # below stay on the step's own (Nnn) state, matching rows 8/9 (already
        # conformant, ``_step_impl`` never widens what reads Nbb here beyond
        # this one tendency's tracer argument -- Rule 1d guard).  ``None``
        # (every other caller) ⇒ bit-identical.
        _T_gm_in = T_mid if _ldf_state is None else _ldf_state[0]
        _S_gm_in = S_mid if _ldf_state is None else _ldf_state[1]

        # GM/Redi isopycnal mixing (if configured)
        k33_implicit = None  # vertical isoneutral diffusivity K_33 for the
        #                      implicit tracer solve (set below iff implicit_K33).
        if self.config.gm_redi is not None:
            gm_cfg = self.config.gm_redi
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
                    )
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
                        E_in, _grid, self.z_coord, gm_cfg,
                        eos=self.config.eos, eos_linear=self.config.eos_linear,
                        mask=lm,
                        rho_0=self.config.constants.rho_0,
                        g=self.config.constants.g,
                        omega=self.config.constants.Omega,
                        r_earth=self.config.constants.R_earth,
                    )
                    # Barotropic production B_T = ∫κ_u|∇h u_h|² dz (Eq. 3)
                    # from the start-of-step velocities (the same time level
                    # that advects E below).
                    prod_bt = geometric_barotropic_production(
                        state.u.data, state.v.data, _grid, geom.kappa_u,
                        dz_geom, lm, state.u_mask.data, state.v_mask.data,
                        z_coord=self.z_coord,
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
                        E, _grid, self.z_coord, gm_cfg,
                        eos=self.config.eos, eos_linear=self.config.eos_linear,
                        mask=lm,
                        rho_0=self.config.constants.rho_0,
                        g=self.config.constants.g,
                        omega=self.config.constants.Omega,
                        r_earth=self.config.constants.R_earth,
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
            if gm_cfg.implicit_K33:
                # ``_T_gm_in``/``_S_gm_in`` (Nbb under nemo_mlf, else T_mid):
                # this hoisted density/jacobian is reused by BOTH the K33
                # diagonal AND the tendency call below, so it MUST be built
                # from the SAME tracer source the tendency call reads -- else
                # K33 and the isoneutral flux it augments would disagree on
                # which time level they came from.
                _gm_dens_jac = gm_redi_density_and_jacobian(
                    _T_gm_in, _S_gm_in, state_new.eta.data,
                    state_new.H_bathy.data,
                    _grid, self.z_coord,
                    eos=self.config.eos, eos_linear=self.config.eos_linear,
                    mask=state.land_mask.data,
                    rho_0=self.config.constants.rho_0,
                    g=self.config.constants.g,
                    eos_depth=getattr(self.config, "eos_depth", "insitu"),
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
                _T_gm_in, _S_gm_in, state_new.eta.data, state_new.H_bathy.data,
                _grid, self.z_coord, gm_cfg,
                eos=self.config.eos, eos_linear=self.config.eos_linear,
                mask=state.land_mask.data,
                u_mask=state.u_mask.data,
                v_mask=state.v_mask.data,
                rho_0=self.config.constants.rho_0, g=self.config.constants.g,
                # self.config.omega (NOT self.config.constants.Omega): the
                # same split as g's momentum-path/GM-Redi-path divergence
                # elsewhere in this function -- self.config.omega is the
                # field dino_lat_lon_model_config actually threads
                # cfg.omega into (from_flat top-level field); constants.Omega
                # stays the unused NamedTuple default. See LatLonCGridOceanConfig
                # docstring / #1226.
                omega=self.config.omega,
                kappa_gm_override=kappa_gm_override,
                kappa_redi_override=kappa_redi_override,
                kappa_redi_v_override=kappa_redi_v_override,
                density_jacobian=_gm_dens_jac,
                return_bolus_transport=_want_bolus,
                dt=dt,
                eos_depth=getattr(self.config, "eos_depth", "insitu"),
            )
            if _want_bolus:
                dT_gm, dS_gm, _bolus = _gm_out
                if _bolus is not None:
                    mass_flux_u_tr, mass_flux_v_tr, w_baro_tr = (
                        add_bolus_to_advecting_flux(
                            _bolus, mass_flux_u, mass_flux_v,
                            u_mask_3d_tracer, v_mask_3d_tracer, _grid,
                            self.z_coord,
                        )
                    )
            else:
                dT_gm, dS_gm = _gm_out
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
                    _T_gm_in, _S_gm_in, state_new.eta.data,
                    state_new.H_bathy.data,
                    _grid, self.z_coord, gm_cfg,
                    eos=self.config.eos, eos_linear=self.config.eos_linear,
                    mask=state.land_mask.data,
                    rho_0=self.config.constants.rho_0, g=self.config.constants.g,
                    kappa_redi_override=kappa_redi_override,
                    kappa_redi_v_override=kappa_redi_v_override,
                    density_jacobian=_gm_dens_jac,
                    # #1226: the SAME wall masks the tendency dispatcher uses,
                    # so the nemo_native K33 slopes/masks are bit-identical to
                    # the explicit operator's (staircase-aware; the K33-side
                    # 2-D rebuild was flat-bottom-only).
                    u_mask=state.u_mask.data,
                    v_mask=state.v_mask.data,
                    dt=dt,
                    eos_depth=getattr(self.config, "eos_depth", "insitu"),
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

        if self.config.tracer_advection == "som":
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
            _adv = self.config.tracer_advection
            _tti = self.config.tracer_time_integrator
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
                active_3d if getattr(self.config, "tracer_wall_neumann_fill", True)
                else None
            )

            # T+S pair fast path: ONE fused horizontal reconstruction +
            # N-S pad per stage for both tracers (level-axis stack; see
            # compute_advection_flux_div_pair).  Bit-identical to the
            # historical per-tracer calls; non-separable schemes fall
            # back to two single-tracer calls inside the pair helpers.
            # NEMO key_linssh: top-cell concentration/dilution flux (static
            # coordinate flag; see _compute_advection_flux_div).
            _linssh = getattr(self.z_coord, "linear_free_surface", False)
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

                if _tti == "rk3":
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
                        eps = self.config.ab2_epsilon
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
        if getattr(self.config.barotropic, "barotropic_slow_forcing_ab2", False):
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

        # 8. Freshwater forcing (virtual salt flux only)
        #
        # The freshwater eta tendency (F_fw_eta) is now applied inside
        # the barotropic continuity equation (via F_slow_eta), so no
        # post-hoc eta correction is needed.  Only the virtual salt
        # flux remains here, applied to the top layer of S.
        if freshwater is not None and self.config.freshwater_closure != "none":
            dz_0 = h_k_new[..., 0]
            _S_dtype = state_new.S.data.dtype
            # Salinity entering the virtual-salt closure: the fixed scalar
            # S_ref (legacy, bit-identical) or the LOCAL top-cell salinity —
            # NEMO's tra_sbc convention (sfx = emp * sss).  Static config
            # gate; unknown values raise at model construction.  The local
            # field is the post-advection (Now) salinity, column-constant
            # for the runoff-spread channel.
            if getattr(self.config, "freshwater_salinity", "s_ref") == "local":
                _S_fw = state_new.S.data[..., 0].astype(_S_dtype)
            else:
                _S_fw = self.config.S_ref
            from legoesm.ocean.freshwater import resolve_runoff_spread_arg
            _spread_arg = resolve_runoff_spread_arg(self.config)
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
                    freshwater, _S_fw, h_k_new, self.config.rho_0,
                    mask,
                    runoff_spread_m=_spread_arg,
                    area=_grid.area,
                    normalize=bool(getattr(self.config,
                                           "normalize_freshwater", False)),
                )
                S_fw = state_new.S.data + (
                    dt * dS_fw_3d * mask[..., None]).astype(_S_dtype)
            elif getattr(self.config, "normalize_freshwater", False):
                # Global-salt-conserving virtual salt: remove the area-mean of the
                # net freshwater (the OMIP correction) so an unbalanced ∮(P-E+R)
                # does not drift mean salinity.  Shared with the MPAS path.
                from legoesm.ocean.freshwater import normalized_virtual_salt_flux
                dS_fw = normalized_virtual_salt_flux(
                    freshwater, _S_fw, dz_0, self.config.rho_0,
                    _grid.area, mask,
                )
                S_fw = state_new.S.data.at[..., 0].add(
                    (dt * dS_fw * mask).astype(_S_dtype),
                )
            else:
                dS_fw = virtual_salt_flux(
                    freshwater, S_ref=_S_fw, dz_0=dz_0, rho_0=self.config.rho_0,
                )
                # Cast the freshwater contribution to S's dtype so the
                # scatter add does not silently widen on x64 mode (the
                # freshwater struct is built at JAX-default precision in
                # init helpers, which can be f64 while S runs at the
                # storage policy's f32).
                S_fw = state_new.S.data.at[..., 0].add(
                    (dt * dS_fw * mask).astype(_S_dtype),
                )
            state_new = state_new._replace(
                S=state_new.S.replace(data=S_fw),
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
        # NEMO stprk3_stg:440 zub correction (nemo_stage_mean_imposition):
        # capture the post-barotropic-solve depth mean so it can be re-imposed
        # after the implicit vertical solve (which otherwise shifts it).
        _impose_mean = (
            getattr(self.config.barotropic, "nemo_stage_mean_imposition", False)
            and _apply_implicit_vmix)
        if _impose_mean:
            _u_mean_baro, _v_mean_baro = self._fixed_depth_means(state_new)

        tke_new = None
        if self.config.implicit_vertical_mixing and _apply_implicit_vmix:
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
                _tke_source = self._assemble_tke_source(state, state_new, tend)
                # NEMO eosbn2 Nnow sequencing: sample the diffusivity-stage
                # N² on the STEP-ENTRY (before-advection) T/S when the flag is
                # set. ``state`` here is the step-entry state (never rebound;
                # ``state_new`` is the working copy). None ⇒ BIT-IDENTICAL.
                _n2_tracers = self._n2_before_advection_tracers(state)
                _n2_tracers_before = self._n2_nemo_before_tracers(state)
                state_new, tke_new = self._apply_implicit_vertical_mixing(
                    state_new, dt, surface_forcing,
                    K_v_phys=tend.K_v, A_v_phys=tend.A_v,
                    K33_iso=k33_implicit, dt_mom=dt_mom,
                    surface_tracer_forcing=tend.surface_tracer_forcing,
                    tracer_source=tend.tracer_source,
                    tke_old=_tke_old, tke_source=_tke_source, return_tke=True,
                    grid=_grid, n2_tracers=_n2_tracers,
                    n2_tracers_before=_n2_tracers_before,
                    # NEMO e3w(Kmm) divisor (#1226 W1): state_new is the
                    # post-update AFTER state; state.eta is NOW. No-op when
                    # implicit_vmix_e3t_now_divisor is off.
                    eta_now=state.eta.data,
                )
            else:
                _n2_tracers = self._n2_before_advection_tracers(state)
                _n2_tracers_before = self._n2_nemo_before_tracers(state)
                state_new = self._apply_implicit_vertical_mixing(
                    state_new, dt, surface_forcing,
                    K_v_phys=tend.K_v, A_v_phys=tend.A_v,
                    K33_iso=k33_implicit, dt_mom=dt_mom,
                    surface_tracer_forcing=tend.surface_tracer_forcing,
                    tracer_source=tend.tracer_source,
                    grid=_grid, n2_tracers=_n2_tracers,
                    n2_tracers_before=_n2_tracers_before,
                    # NEMO e3w(Kmm) divisor (#1226 W1): see the sibling call.
                    eta_now=state.eta.data,
                )
        if tke_new is not None:
            # Veros order (integrate_tke): the implicit solve writes
            # tke[taup1] FIRST, then the superbee-advection AB2 increment is
            # added to it (tke.py:315-323). Advects the carried state.tke
            # (tke[tau]) by the pre-step state.u/v (u[tau]); dt here is the
            # TRACER dt (Veros dt_tracer). Static gate ⇒ default adds no ops.
            if self._tke_advection_active():
                tke_new, _dtke_field = self._apply_tke_advection(
                    state, tke_new, dt, grid=_grid)
                state_new = state_new._replace(dtke=_dtke_field)
            state_new = state_new._replace(
                tke=Field(data=tke_new, name="tke",
                          dims=("lat", "lon", "level"), units="m^2/s^2"),
            )

        if _impose_mean:
            # NEMO stprk3_stg.F90:440: uu += (uu_b(Kaa) − Σ e3u_0·uu·r1_hu_0)
            # ·umask — the 3D velocity's depth mean is REPLACED by the
            # barotropic solution after the implicit solve, uniformly over
            # the column. Sign convention: an ADDITIVE column-uniform shift,
            # so the baroclinic deviation u′ is untouched (budget: the
            # depth-integral becomes exactly the barotropic transport).
            _u_mean_now, _v_mean_now = self._fixed_depth_means(state_new)
            _du = (_u_mean_baro - _u_mean_now)[..., jnp.newaxis]
            _dv = (_v_mean_baro - _v_mean_now)[..., jnp.newaxis]
            _um = state.u_mask.data[..., jnp.newaxis]
            _vm = state.v_mask.data[..., jnp.newaxis]
            state_new = state_new._replace(
                u=state_new.u.replace(data=state_new.u.data + _du * _um),
                v=state_new.v.replace(data=state_new.v.data + _dv * _vm),
            )

        # 9. Conservation fixers
        if self.config.use_conservation_fixer and _apply_implicit_vmix:
            state_new = ocean_conservation_fixer(
                state_new, state, _grid, self.z_coord, self.config,
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

        state_new = cast_pytree(state_new, None, "storage", allow_downcast=True)
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
            _tke_src = (self._assemble_tke_source(state, state_new, tend)
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
            return state_new, (tend.K_v, tend.A_v, k33_implicit,
                               tend.surface_tracer_forcing, _tke_src,
                               _diss_incr, tend.tracer_source)
        return state_new

    def _fixed_depth_means(self, st):
        """Thickness-weighted depth means of u, v on FIXED reference
        thicknesses (NEMO ``e3u_0``/``r1_hu_0``, the linssh convention used by
        the stprk3_stg:440 zub correction). Face thicknesses by the min-rule,
        matching the barotropic solver's depth average
        (``_depth_average_to_faces``) so imposition restores exactly the mean
        the barotropic solve set."""
        from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
            barotropic_depth_average_to_faces as _depth_average_to_faces,
        )
        from legoesm.ocean.vertical import compute_layer_thickness
        h_k = compute_layer_thickness(
            jnp.zeros_like(st.eta.data), st.H_bathy.data, self.z_coord,
            min_water_column_m=self.config.min_water_column_m,
        ).astype(st.u.data.dtype)
        return _depth_average_to_faces(
            st.u.data, st.v.data, h_k,
            jnp.asarray(self.config.min_water_column_m, dtype=st.u.data.dtype),
            st.land_mask.data, st.u_mask.data, st.v_mask.data, self.grid)

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

    def _n2_before_advection_tracers(self, entry_state):
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
        vmix = getattr(getattr(self.config, "physics", None),
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

    def _n2_nemo_before_tracers(self, entry_state):
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
        vmix = getattr(getattr(self.config, "physics", None),
                       "vertical_mixing", None)
        if vmix is None or vmix.scheme != "tke":
            return None
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

    def _tke_bottom_dirichlet(self, state):
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
        vmix = getattr(getattr(self.config, "physics", None),
                       "vertical_mixing", None)
        if vmix is None or vmix.scheme != "tke":
            return None
        if not getattr(vmix.tke, "bottom_tke_bc", False):
            return None
        if not isinstance(self.z_coord, OceanPartialCellCoordinate):
            raise ValueError(
                "vertical_mixing.tke.bottom_tke_bc=True requires a "
                "partial-cell z-coordinate (bottom_level) — the flat-bottom "
                "case is not covered.")
        from legoesm import constants
        from legoesm.ocean.dynamics.ocean_tendency_common import (
            nemo_effective_bottom_drag_r, validate_bottom_drag_scheme,
        )
        _scheme = validate_bottom_drag_scheme(
            str(getattr(self.config.bottom_drag, "bottom_drag_scheme",
                        "legacy")))
        if _scheme == "legacy":
            raise ValueError(
                "vertical_mixing.tke.bottom_tke_bc=True requires a NEMO "
                "bottom_drag_scheme ('nemo_quadratic' or 'nemo_loglayer'), "
                f"got 'legacy'.")
        h_k = self.z_coord.h_partial
        _bl = jnp.maximum(self.z_coord.bottom_level, 0)
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
            cd0=float(self.config.bottom_drag.bottom_drag_cd0),
            cd_max=float(self.config.bottom_drag.bottom_drag_cdmax),
            z0=float(self.config.bottom_drag.bottom_drag_z0),
            ke0=float(self.config.bottom_drag.bottom_drag_ke0),
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
        um3, vm3 = compute_face_masks_3d(self.z_coord.is_active, self.grid)
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
        ssmask = _at_bottom(self.z_coord.is_active.astype(h_k.dtype))
        return ssmask * nemo_bottom_tke_dirichlet(r_t, u_sum, v_sum, vmix.tke)

    def _tke_bottom_level(self):
        """Per-column T-point bottom-cell index for the T15-exact bottom TKE
        Dirichlet placement, or None.

        Static Python predicate mirroring ``_tke_bottom_dirichlet``'s gate:
        only meaningful together with a held bottom value, and only when a
        partial-cell coordinate actually carries a per-column
        ``bottom_level`` (the flat-bottom / pure z-star case has none —
        ``bottom_dirichlet``'s unconditional last-row pin is already exact
        there, so ``None`` here keeps that path BIT-IDENTICAL).
        """
        vmix = getattr(getattr(self.config, "physics", None),
                       "vertical_mixing", None)
        if vmix is None or vmix.scheme != "tke":
            return None
        if not getattr(vmix.tke, "bottom_tke_bc", False):
            return None
        if not isinstance(self.z_coord, OceanPartialCellCoordinate):
            return None
        return self.z_coord.bottom_level

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

    def _apply_tke_advection(self, state, tke_new, dt, *, grid=None):
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
        from legoesm.ocean.advection import (
            wgrid_advection_tendency_latlon_cgrid,
        )

        _grid = grid if grid is not None else self.grid
        tke_cfg = self.config.physics.vertical_mixing.tke
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
        if isinstance(self.z_coord, OceanPartialCellCoordinate):
            _um3, _vm3 = compute_face_masks_3d(
                self.z_coord.is_active, _grid)
            u_adv = u_adv * _um3.astype(u_adv.dtype)
            v_adv = v_adv * _vm3.astype(v_adv.dtype)
            _wet_if_adv = self.z_coord.is_active.astype(dtype)[..., 1:]
            tke_tau = tke_tau * _wet_if_adv
        dtke_now = wgrid_advection_tendency_latlon_cgrid(
            tke_tau, u_adv, v_adv, _grid,
            jnp.asarray(self.z_coord.dz_ref), dt, lm,
            state.u_mask.data, state.v_mask.data,
        )
        dtke_now = jax.lax.convert_element_type(dtke_now, dtype)
        if _wet_if_adv is not None:
            dtke_now = dtke_now * _wet_if_adv
        dtke_prev = (state.dtke.data.astype(dtype)
                     if state.dtke is not None else jnp.zeros_like(dtke_now))
        eps = self.config.ab2_epsilon
        # (#517 item 8: shared ab2_blend; eps verbatim → bit-identical.)
        tke_out = tke_new + dt * ab2_blend(dtke_now, dtke_prev, eps)
        dtke_field = Field(data=dtke_now, name="dtke",
                           dims=("lat", "lon", "level"), units="m^2/s^3")
        return tke_out, dtke_field

    def _assemble_tke_source(self, state, state_new, tend):
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
        tke_cfg = self.config.physics.vertical_mixing.tke
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
            E, _grid, self.z_coord, gm_cfg,
            eos=self.config.eos, eos_linear=self.config.eos_linear,
            mask=lm,
            rho_0=self.config.constants.rho_0,
            g=self.config.constants.g,
            omega=self.config.constants.Omega,
            r_earth=self.config.constants.R_earth,
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
        if isinstance(self.z_coord, OceanPartialCellCoordinate):
            _um3, _vm3 = compute_face_masks_3d(
                self.z_coord.is_active, _grid)
            _u_lvl = _u_lvl * _um3.astype(_u_lvl.dtype)
            _v_lvl = _v_lvl * _vm3.astype(_v_lvl.dtype)
            _wet_if_eke = self.z_coord.is_active.astype(dtype)[..., 1:]
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
            state.eta.data, state.H_bathy.data, self.z_coord,
        )
        # dz metrics in the field dtype so the implicit solve stays consistent
        # (dz_ref is f64 while the eddy-energy field runs at the storage policy's
        # dtype — cast to E's dtype to avoid a f64->f32 scatter cast).
        dz_cell = (self.z_coord.dz_ref * J_cell[..., jnp.newaxis]).astype(dtype)
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
            A_v_phys, jnp.asarray(self.config.A_v, dtype=dtype), nlev,
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
                _grid, self.z_coord, gm_cfg, _kappa_gm_src,
                eos=self.config.eos, eos_linear=self.config.eos_linear,
                mask=lm,
                rho_0=self.config.constants.rho_0, g=self.config.constants.g,
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
                eos=self.config.eos, eos_linear=self.config.eos_linear,
                mask=lm, u_mask=state.u_mask.data, v_mask=state.v_mask.data,
                rho_0=self.config.constants.rho_0, g=self.config.constants.g,
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
                    _grid, self.z_coord, gm_cfg, _kappa_gm_src,
                    want_skew=True, want_iso=False, kappa_redi_w=None,
                    **_signed_kwargs,
                )
                _, neg_iso = compute_realized_signed_conversions(
                    T_mid, S_mid, state_new.eta.data, state_new.H_bathy.data,
                    _grid, self.z_coord, gm_cfg, kappa_gm_override,
                    want_skew=False, want_iso=True, kappa_redi_w=None,
                    **_signed_kwargs,
                )
            else:
                neg_skew, neg_iso = compute_realized_signed_conversions(
                    T_mid, S_mid, state_new.eta.data, state_new.H_bathy.data,
                    _grid, self.z_coord, gm_cfg, _kappa_gm_src,
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
        surface_tracer_forcing=None,
        tracer_source=None,
        do_tracers: bool = True,
        do_momentum: bool = True,
        tke_old=None,
        tke_source=None,
        return_tke: bool = False,
        K_diss_v_w=None,
        return_K_diss_v: bool = False,
        grid=None,
        n2_tracers=None,
        n2_tracers_before=None,
        eta_now=None,
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
        if dt_mom is None:
            dt_mom = dt
        _grid = grid if grid is not None else self.grid
        if return_K_diss_v and return_tke:
            raise ValueError(
                "_apply_implicit_vertical_mixing: return_K_diss_v is for the "
                "momentum-only call (return_tke must be False).")
        tke_new = None
        _post_mixing = self._tke_post_mixing_active()
        _tke_ctx = None
        from legoesm.ocean.physics.vertical_mixing import (
            implicit_vertical_diffusion_ocean_batched, build_dz_half,
            compute_vertical_K_profiles,
        )
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
            K_v_cell = K_v_phys + jnp.asarray(self.config.K_v, dtype=dtype)
            A_v_cell = A_v_phys + jnp.asarray(self.config.A_v, dtype=dtype)
            _phys_cfg = self.config.physics
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
                    state, self.z_coord, self.config.physics,
                    _phys_cfg.vertical_mixing.iwm,
                    eos_fn=_mk_eos(eos=self.config.eos,
                                   eos_linear=self.config.eos_linear),
                    iwm_fields=self._iwm_forcing,
                ).astype(dtype)
                K_v_cell = K_v_cell + _K_iwm
                A_v_cell = A_v_cell + _K_iwm
        else:
            # Fallback: recompute K profiles (expensive for KPP).
            physics_config = self.config.physics
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
                           "squared_centered") == "nemo_face_native"
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
                eos=self.config.eos, eos_linear=self.config.eos_linear,
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
            if _tke_prognostic:
                K_v_cell, A_v_cell, tke_new = compute_vertical_K_profiles(
                    cc_state, self.z_coord, surface_forcing, physics_config,
                    A_v_background=float(self.config.A_v),
                    K_v_background=float(self.config.K_v),
                    eos_fn=_vmix_eos_fn,
                    tke_old=tke_old, dt_tke=dt_mom,
                    tke_source=tke_source, return_tke=True,
                    # T-point latitudes [deg] for the NEMO etau_htau_mode=
                    # "latitude" penetration profile (unused otherwise).
                    # 2-D lat_T (exact on the tripole, where rows curve and
                    # the legacy 1-D grid.lat is a row mean) — a 1-D (n_lat,)
                    # array cannot right-broadcast against the (n_lat, n_lon,
                    # nlev-1) columns inside nemo_etau_injection.
                    lat_deg=jnp.degrees(self.grid.lat_T),
                    iwm_fields=self._iwm_forcing,
                    n2_tracers=n2_tracers,
                    tke_bottom_dirichlet=self._tke_bottom_dirichlet(state),
                    tke_bottom_level=self._tke_bottom_level(),
                    n2_tracers_before=n2_tracers_before,
                )
                if _post_mixing:
                    # Phase 1 only (Veros set_tke_diffusivities from the
                    # carried tke[tau]): the third slot is the post-mixing
                    # CONTEXT, not an updated TKE — the budget is solved
                    # below, AFTER the tracer solve.
                    _tke_ctx, tke_new = tke_new, None
            else:
                K_v_cell, A_v_cell = compute_vertical_K_profiles(
                    cc_state, self.z_coord, surface_forcing, physics_config,
                    A_v_background=float(self.config.A_v),
                    K_v_background=float(self.config.K_v),
                    eos_fn=_vmix_eos_fn,
                    lat_deg=jnp.degrees(self.grid.lat_T),
                    iwm_fields=self._iwm_forcing,
                    n2_tracers=n2_tracers,
                    n2_tracers_before=n2_tracers_before,
                )

        # dz at cell centers (jacobian-corrected so the eta-stretched
        # column heights match the partial-cell / z* layer thicknesses
        # used by every other operator in this step).
        J_cell = compute_ocean_jacobian(
            state.eta.data, state.H_bathy.data, self.z_coord,
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
        if isinstance(self.z_coord, OceanPartialCellCoordinate):
            dz_cell = self.z_coord.h_partial * J_cell[..., jnp.newaxis]
        else:
            dz_cell = self.z_coord.dz_ref * J_cell[..., jnp.newaxis]
        # Gradient (center-to-center) divisor of the implicit solve.  Default is
        # the midpoint reconstruction 0.5(dz_k+dz_{k+1}); the Veros-faithful slot
        # (config.implicit_vmix_dzw_slot, #428) uses the coordinate's
        # center-to-center spacing dz_half_ref·J = Veros's dzw, which differs from
        # the midpoint on a u_centered z-coordinate.  NO-OP on a midpoint z-star.
        # The NEMO-faithful slot (config.implicit_vmix_e3t_now_divisor, #1226 W1)
        # instead uses the NOW-level (pre-solve) thickness midpoint, matching
        # NEMO's e3w(Kmm) (trazdf.F90:219-220).  Call sites that pass a
        # post-update AFTER state (_leapfrog_step's naa_expl — the DINO
        # kamm_mlf production path; _unsplit_ab2_step's state_corr;
        # _ab2_step's state_ab2; _step_impl's state_new) thread the true NOW
        # eta explicitly via ``eta_now``; the momentum-only friction call
        # passes the step-entry state directly, so its fallback
        # (eta_now=None -> state.eta) IS the NOW eta.  A future call site
        # that passes an AFTER state without eta_now would silently divide by
        # the AFTER thickness — thread eta_now there too.  Static Python
        # bools (feature-gating exception, CLAUDE.md) — config is not traced.
        _dzw_slot = bool(getattr(self.config, "implicit_vmix_dzw_slot", False))
        _e3t_now_slot = bool(
            getattr(self.config, "implicit_vmix_e3t_now_divisor", False))
        # Cell-centered NOW thickness (only computed / used under the e3t_now
        # slot; also feeds the u/v-face divisor below so tracer and momentum
        # solves share one NOW-eta evaluation).
        e3t_now = None
        if _dzw_slot:
            dz_half_cell = (self.z_coord.dz_half_ref
                            * J_cell[..., jnp.newaxis]).astype(dz_cell.dtype)
        elif _e3t_now_slot:
            _eta_now = eta_now if eta_now is not None else state.eta.data
            e3t_now = compute_layer_thickness(
                _eta_now, state.H_bathy.data, self.z_coord,
                min_water_column_m=self.config.min_water_column_m)
            dz_half_cell = build_dz_half(e3t_now).astype(dz_cell.dtype)
        else:
            dz_half_cell = build_dz_half(dz_cell)

        mask_3d = state.land_mask.data[..., jnp.newaxis]

        # Partial-cell dry-interface guard: no implicit flux through the
        # seafloor.  K/A at interface k couple cells k and k+1; where cell k+1
        # is below the seafloor the diffusivity must be EXACTLY zero (Veros
        # maskW), or the solve mixes the bottom wet cell with the T=S=0 (or
        # u=0) rock cells — incl. the K33 isoneutral diagonal and the A_v/K_v
        # config backgrounds, which are NOT covered by the K-profile-level
        # masking in compute_vertical_K_profiles.  Pure z-star: no-op.
        if isinstance(self.z_coord, OceanPartialCellCoordinate):
            _wet_if_vmix = self.z_coord.is_active.astype(
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
                getattr(getattr(self.config, "physics", None),
                        "vertical_mixing", None), "ddm", None)
            if _ddm_cfg is not None and _ddm_cfg.enabled:
                from legoesm.ocean.physics.vertical_mixing.k_profiles import (
                    ddm_K_profile,
                )
                from legoesm.ocean.eos import make_eos_fn as _mk_eos_ddm
                _avt_ddm, _avs_ddm = ddm_K_profile(
                    state, self.z_coord, self.config.physics, _ddm_cfg,
                    eos_fn=_mk_eos_ddm(eos=self.config.eos,
                                       eos_linear=self.config.eos_linear),
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
            # IMPLICIT surface TRACER forcing (Veros placement): add dt·S_surf
            # (masked) to the solve INPUT so the backward-Euler tridiagonal solve
            # realises ``(I − dt·L)·X_new = X_old + dt·S_surf`` at weight 1.0.  dt
            # here is dt_tracer (the tracer timestep), matching Veros's
            # ``dt_tracer·forc/dz[surface]`` RHS source.  No-op when None.
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
        if do_momentum and getattr(self.config, "surface_stress_implicit",
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
                surface_forcing, state.u.data.dtype, self.z_coord, J_cell,
                _grid)
            if _sfx is not None:
                _tau_i_u, _tau_j_v, _dz0u, _dz0v = _sfx
                _r0 = jnp.asarray(self.config.constants.rho_0,
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
                    self.z_coord.is_active, _grid)
                A_v_u = A_v_u * _act_u3.astype(A_v_u.dtype)[..., 1:]
                A_v_v = A_v_v * _act_v3.astype(A_v_v.dtype)[..., 1:]
            if _dzw_slot:
                # Veros dzw at u/v-faces (#428): dz_half_ref·J interpolated to the
                # faces with the SAME interp that built the control volumes
                # dz_u/dz_v (J is level-independent, so dz_u = dz_ref·J_u and the
                # gradient slot dz_half_ref·J_u stays consistent with it).
                J_u = interp_cell_to_uface(J_cell[..., jnp.newaxis])
                J_v = interp_to_v_points(J_cell[..., jnp.newaxis], _grid)
                dz_half_u = (self.z_coord.dz_half_ref * J_u).astype(dz_u.dtype)
                dz_half_v = (self.z_coord.dz_half_ref * J_v).astype(dz_v.dtype)
            elif _e3t_now_slot:
                # NEMO e3w(Kmm) at u/v-faces (#1226 W1): interpolate the SAME
                # NOW-level thickness (e3t_now, cell-centered above) to the
                # faces with the SAME interps used for dz_u/dz_v, matching the
                # dzw-slot sibling's face-consistency pattern.
                e3t_now_u = interp_cell_to_uface(e3t_now)
                e3t_now_v = interp_to_v_points(e3t_now, _grid)
                dz_half_u = build_dz_half(e3t_now_u).astype(dz_u.dtype)
                dz_half_v = build_dz_half(e3t_now_v).astype(dz_v.dtype)
            else:
                dz_half_u = build_dz_half(dz_u)
                dz_half_v = build_dz_half(dz_v)
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
            do_momentum and getattr(self.config, "zdf_baroclinic_only", False))
        if _zdf_baroclinic_only:
            _u_bt_mean = depth_mean(
                u_solve_in, dz_u, self.config.min_water_column_m,
                keepdims=True)
            _v_bt_mean = depth_mean(
                v_solve_in, dz_v, self.config.min_water_column_m,
                keepdims=True)
            u_solve_in = u_solve_in - _u_bt_mean
            v_solve_in = v_solve_in - _v_bt_mean

        # zdf_drag_in_matrix: NEMO's semi-implicit bottom friction goes INTO
        # the tridiagonal diagonal at each face-column's deepest wet cell
        # (dynzdf.F90:293-305) instead of the explicit RHS kick
        # (_bc_bottom_drag, disabled at the tendency stage when this flag is
        # on — see the ocean_pe_latlon_cgrid single-owner guard).  Sign:
        # NEMO's rCdU_bot <= 0 and the diagonal SUBTRACTS the SUM of the two
        # T-point rates (zwd -= zDt_2*(rCdU_bot(i+1,j)+rCdU_bot(i,j))/e3u),
        # which ADDS positive definiteness (damping); legoESM's r_eff =
        # -rCdU_bot >= 0 is the 0.5-AVERAGE of those same two rates, so
        # extra_diag = +2*dt_mom*r_eff/h at the bottom cell reproduces
        # NEMO's sum (see the factor-of-2 comment at the extra_diag_u/v
        # assignment below).  This flag requires a NEMO bottom-drag scheme,
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
        if do_momentum and getattr(self.config, "zdf_drag_in_matrix", False):
            from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
                nemo_bottom_drag_rate_faces,
            )
            _r_eff_u, _r_eff_v, _is_bot_u, _is_bot_v = (
                nemo_bottom_drag_rate_faces(
                    state.u.data, state.v.data, dz_cell, self.z_coord,
                    self.config, _grid))
            _r_eff_u = _r_eff_u.astype(state.u.data.dtype)
            _r_eff_v = _r_eff_v.astype(state.v.data.dtype)
            # NEMO dynzdf.F90:293-296: zwd(iku) -= zDt_2*(rCdU_bot(i+1,j)
            # + rCdU_bot(i,j))/e3u(iku) -- a SUM of the two T-point rates
            # (no 1/2), with zDt_2 = the physical timestep (not the 2*dt
            # leapfrog form despite the name). ``_r_eff_{u,v}`` is
            # ``nemo_bottom_drag_rate_faces``'s 0.5*(...) AVERAGE of those
            # same two T-point rates (the shared helper used elsewhere for
            # the RHS drag kick), so reproducing NEMO's sum from the
            # average requires the explicit factor of 2 here.
            extra_diag_u = (
                2.0 * dt_mom * _r_eff_u[..., jnp.newaxis]
                / jnp.maximum(dz_u, 1e-10) * _is_bot_u)
            extra_diag_v = (
                2.0 * dt_mom * _r_eff_v[..., jnp.newaxis]
                / jnp.maximum(dz_v, 1e-10) * _is_bot_v)
            if _zdf_baroclinic_only:
                # NEMO dynzdf.F90:156-159: puu(Krhs) += zDt_2*(rCdU_bot sum)
                # * uu_b(Kaa)/e3u(iku), with rCdU_bot <= 0 in NEMO's
                # convention -- so this RHS term is NEGATIVE (it damps the
                # barotropic bottom velocity uu_b/vv_b, opposing it, not
                # reinforcing it). legoESM's r_eff = -rCdU_bot >= 0, so the
                # sign-translated term SUBTRACTS from the solve input:
                # the barotropic-mode bottom cell loses ``2*dt_mom*r_eff
                # /e3u * u_bt_mean`` before the solve, matching NEMO's
                # damping direction.
                u_solve_in = u_solve_in - (
                    2.0 * dt_mom * _r_eff_u[..., jnp.newaxis]
                    / jnp.maximum(dz_u, 1e-10) * _is_bot_u * _u_bt_mean)
                v_solve_in = v_solve_in - (
                    2.0 * dt_mom * _r_eff_v[..., jnp.newaxis]
                    / jnp.maximum(dz_v, 1e-10) * _is_bot_v * _v_bt_mean)

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
            and not getattr(self.config, "zdf_drag_in_matrix", False)
        )
        # Double-diffusion salinity diffusivity: K_v (heat) + (avs - avt).
        # ``dK_ddm_salt is None`` (ddm off) ⇒ K_s_cell IS K_v_cell (same
        # object) ⇒ the shared-K pair fast path stays BYTE-IDENTICAL.
        K_s_cell = K_v_cell if dK_ddm_salt is None else (K_v_cell + dK_ddm_salt)
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
                # T and S share the IDENTICAL tridiagonal matrix (same
                # K_v_cell incl. any K33_iso fold + partial-cell wet
                # mask, same dz/dz_half/dt), so the pair solve factors
                # it ONCE — bit-identical outputs, one fewer
                # coefficient build + forward-factor sweep (vmix is
                # memory-bandwidth-bound; this REMOVES traffic where
                # field-batching ADDED it).  LEGOESM_VMIX_TSPAIR=0
                # restores the two separate solves (trace-time switch,
                # same caveat as above).
                if (dK_ddm_salt is None
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
                u_new = implicit_vertical_diffusion_ocean(
                    u_solve_in, A_v_u, dz_u, dz_half_u, dt_mom,
                    extra_diag=extra_diag_u,
                )
                v_new = implicit_vertical_diffusion_ocean(
                    v_solve_in, A_v_v, dz_v, dz_half_v, dt_mom,
                    extra_diag=extra_diag_v,
                )
        if do_tracers:
            T_new = jnp.where(mask_3d > 0.5, T_new, state.T.data)
            S_new = jnp.where(mask_3d > 0.5, S_new, state.S.data)
        if do_momentum:
            if _zdf_baroclinic_only:
                # Re-add the SAME depth mean that was subtracted before the
                # solve (dynzdf.F90's barotropic component re-enters via
                # mlf_baro_corr AFTER dyn_zdf) — exactly conservative at
                # A_v=0 (test 3: strip + re-add round-trips to the input).
                u_new = u_new + _u_bt_mean
                v_new = v_new + _v_bt_mean
            u_new = jnp.where(u_mask_3d > 0.5, u_new, state.u.data)
            v_new = jnp.where(v_mask_3d > 0.5, v_new, state.v.data)
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
            tke_cfg = self.config.physics.vertical_mixing.tke
            T_n2, S_n2 = T_new, S_new
            if isinstance(self.z_coord, OceanPartialCellCoordinate):
                # Same sub-seafloor guard the phase-1 N² used (k_profiles):
                # extend the deepest ACTIVE T/S downward so the seafloor
                # interface reads neutral, not the T=S=0 rock fill.
                from legoesm.ocean.vertical import extrapolate_below_seafloor
                T_n2 = extrapolate_below_seafloor(T_n2, self.z_coord)
                S_n2 = extrapolate_below_seafloor(S_n2, self.z_coord)
            # Veros recomputes Nsqr[taup1] at the STATIC reference pressures
            # (press = abs(zt)); reuse the phase-1 cell pressures (the
            # hydrostatic p of the pre-solve state — the mixing step does
            # not move the pressure field).
            N2_post = compute_buoyancy_frequency_adiabatic(
                T_n2, S_n2, _tke_ctx.p_cell,
                self.z_coord.dz_ref, J_cell,
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
        if return_K_diss_v:
            return state_out, K_diss_v_w
        if return_tke:
            return state_out, tke_new
        return state_out

    def step(self, state: LatLonCGridOceanState, dt: float,
             freshwater=None, surface_forcing=None,
             sponge=None, *, grid=None,
             vertex_mask=None, t_seconds=None,
             external_tracer_rate=None) -> LatLonCGridOceanState:
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
        return self._step_jitted(
            state, dt, freshwater, surface_forcing, sponge,
            grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds,
            external_tracer_rate=external_tracer_rate)

    @partial(jax.jit, static_argnums=(0,))
    def _step_jitted(self, state: LatLonCGridOceanState, dt: float,
                     freshwater=None, surface_forcing=None,
                     sponge=None, *, grid=None,
                     vertex_mask=None, t_seconds=None,
                     external_tracer_rate=None) -> LatLonCGridOceanState:
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
                                        t_seconds=t_seconds)
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
                  t_seconds=None) -> LatLonCGridOceanState:
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
            t_seconds=t_seconds)
        _tke_prog = self._tke_prognostic_active()
        _tke_old = (state.tke.data if (_tke_prog and state.tke is not None)
                    else None)
        eps = self.config.ab2_epsilon
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
        if isinstance(self.z_coord, OceanPartialCellCoordinate):
            _au3, _av3 = compute_face_masks_3d(
                self.z_coord.is_active, _grid)
            u_mask3 = u_mask3 * _au3.astype(u_mask3.dtype)
            v_mask3 = v_mask3 * _av3.astype(v_mask3.dtype)
            mask3 = mask3 * self.z_coord.is_active.astype(mask3.dtype)

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
            state.eta.data, state.H_bathy.data, self.z_coord,
            min_water_column_m=self.config.min_water_column_m)
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
        _pflow = getattr(self.config, "prescribed_flow", None)
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
        if self.config.implicit_vertical_mixing:
            dt_mom = dt / self.config.dt_mom_ratio
            if self.config.momentum_friction_additive:
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
                )
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
                    n2_tracers=self._n2_before_advection_tracers(state),
                    # NEMO e3w(Kmm) divisor (#1226 W1): state_ab2 is the
                    # post-AB2 AFTER state; state.eta is NOW (un-rebound
                    # _ab2_step parameter). No-op when the flag is off.
                    eta_now=state.eta.data,
                )
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
                    n2_tracers=self._n2_before_advection_tracers(state),
                    # NEMO e3w(Kmm) divisor (#1226 W1): see the sibling
                    # additive-friction tracer call above.
                    eta_now=state.eta.data,
                )
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
                    state, tke_new_ab2, dt, grid=_grid)
                state_ab2 = state_ab2._replace(dtke=_dtke_field)
            state_ab2 = state_ab2._replace(
                tke=Field(data=tke_new_ab2, name="tke",
                          dims=("lat", "lon", "level"), units="m^2/s^2"),
            )

        # --- Conservation fixer once, on the final state ---
        if self.config.use_conservation_fixer:
            state_ab2 = ocean_conservation_fixer(
                state_ab2, state, _grid, self.z_coord, self.config,
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

    def _leapfrog_step(self, state: LatLonCGridOceanState, dt: float,
                       freshwater=None, surface_forcing=None, sponge=None,
                       *, grid=None, vertex_mask=None,
                       t_seconds=None,
                       external_tracer_rate=None) -> LatLonCGridOceanState:
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
        from legoesm.ocean.state import Field
        _grid = grid if grid is not None else self.grid

        # --- FIRST step: forward-Euler start (NEMO l_1st_euler), no RA filter.
        #     Populate Nbb with the pre-step now-fields for the next step.
        if state.u_before is None:
            # NEMO's cold-start Euler step does NOT run with an undefined
            # before-level: istate.F90:97-99/135-137 sets Kmm := Kbb (ts/uu/vv
            # copied onto BOTH time-level array slots) before stp_MLF is ever
            # called, so Nbb==Nnn identically on this very first step; combined
            # with stpmlf.F90:114-117 (l_1st_euler -> rDt=rn_Dt) this makes the
            # leap-frog combine degenerate exactly to forward-Euler. The
            # rn2b/Burchard-shear consumers inside _step_impl (nemo_before N²,
            # nemo_burchard shear production) read entry_state.T_before/
            # S_before/u_before/v_before unconditionally, so they need this same
            # before==now seed on THIS call only -- a LOCAL copy, not written
            # back onto ``state``/``naa`` below, which must keep the ``None``
            # sentinel so this branch still fires (single-dt, no RA filter) and
            # the real Nbb seed at :6656 still runs from the true pre-step now-
            # fields. A bridged/restart state never reaches this branch (its
            # u_before is already populated), so this seed only ever applies to
            # a genuine from-rest / no-history state -- exactly NEMO's case.
            _entry = state._replace(
                u_before=state.u, v_before=state.v, T_before=state.T,
                S_before=state.S, eta_before=state.eta,
            )
            naa = self._step_impl(
                _entry, dt, freshwater=freshwater,
                surface_forcing=surface_forcing, sponge=sponge, grid=_grid,
                vertex_mask=vertex_mask, t_seconds=t_seconds)
            naa = naa._replace(
                u_before=state.u, v_before=state.v, T_before=state.T,
                S_before=state.S, eta_before=state.eta,
            )
            if getattr(self.config, "barotropic_forcing_centred", False):
                naa = naa._replace(
                    **_seed_centred_forcing_carry(
                        surface_forcing, freshwater, self.config.rho_0,
                        state.land_mask.data))
            return naa

        # --- LEAP-FROG + Asselin.
        rdt = 2.0 * dt
        gamma = self.config.asselin_gamma
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
        if isinstance(self.z_coord, OceanPartialCellCoordinate):
            _au3, _av3 = compute_face_masks_3d(self.z_coord.is_active, _grid)
            u_mask3 = u_mask3 * _au3.astype(u_mask3.dtype)
            v_mask3 = v_mask3 * _av3.astype(v_mask3.dtype)
            mask3 = mask3 * self.z_coord.is_active.astype(mask3.dtype)

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
        # The barotropic mode integrates over rDt=2dt, so its substep count is
        # scaled ×(rDt/dt)=2 to hold the substep length (barotropic CFL) at the
        # FE-path value — see ``_barotropic_substep_scale`` in ``_step_impl``.
        _baro_scale = 2
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
                     tke_source, _diss_incr_nn, tracer_source) = self._step_impl(
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
            _fct_tracer_before=(
                state.T_before.data, state.S_before.data),
            # #1492 DINO surface_tendency_placement="leapfrog_rhs": fold the
            # externally-supplied surface tracer RHS into THIS (Nnn advective)
            # pass only, matching tra_sbc's Nnn-only call — see this method's
            # docstring and the ``step()`` param doc.
            _external_tracer_rate=external_tracer_rate)
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
            _barotropic_substep_scale=_baro_scale)

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
            state.eta.data, state.H_bathy.data, self.z_coord,
            min_water_column_m=self.config.min_water_column_m)
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
        _combine = getattr(self.config, "tracer_combine", "concentration")
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
                state.eta_before.data, state.H_bathy.data, self.z_coord,
                min_water_column_m=self.config.min_water_column_m)
            h_naa = compute_layer_thickness(
                state_expl.eta.data, state.H_bathy.data, self.z_coord,
                min_water_column_m=self.config.min_water_column_m)
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
        #    rDt=2dt on the after-state (NEMO dyn_zdf/tra_zdf, rDt=2dt).
        dt_mom = rdt / self.config.dt_mom_ratio
        _tke_prog = self._tke_prognostic_active()
        _tke_old = (state.tke.data if (_tke_prog and state.tke is not None)
                    else None)
        tke_new = None
        if self.config.implicit_vertical_mixing:
            _res = self._apply_implicit_vertical_mixing(
                naa_expl, rdt, surface_forcing,
                K_v_phys=K_v_phys, A_v_phys=A_v_phys, K33_iso=k33_implicit,
                dt_mom=dt_mom,
                surface_tracer_forcing=surface_tracer_forcing,
                tracer_source=tracer_source,
                tke_old=_tke_old, tke_source=tke_source,
                return_tke=_tke_prog, grid=_grid,
                n2_tracers=self._n2_before_advection_tracers(state),
                n2_tracers_before=self._n2_nemo_before_tracers(state),
                # NEMO e3w(Kmm) divisor (config.implicit_vmix_e3t_now_divisor,
                # #1226 W1): naa_expl.eta is the barotropic AFTER/Kaa level
                # ("h at state_expl.eta is the Kaa thickness" above);
                # state.eta is the Nnn/NOW level (the same eta h_k is built
                # from). This is THE production path for the DINO kamm_mlf
                # recipe (outer_integrator=leapfrog). No-op when the flag is
                # off.
                eta_now=state.eta.data,
            )
            if _tke_prog:
                naa, tke_new = _res
            else:
                naa = _res
        else:
            naa = naa_expl
        if tke_new is not None:
            if self._tke_advection_active():
                tke_new, _dtke_field = self._apply_tke_advection(
                    state, tke_new, rdt, grid=_grid)
                naa = naa._replace(dtke=_dtke_field)
            naa = naa._replace(
                tke=Field(data=tke_new, name="tke",
                          dims=("lat", "lon", "level"), units="m^2/s^2"))

        # 4. Conservation fixer on the final after-state.
        if self.config.use_conservation_fixer:
            naa = ocean_conservation_fixer(
                naa, state, _grid, self.z_coord, self.config)

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
        u_f = _asselin(state.u.data, state.u_before.data, naa.u.data, u_mask3)
        u_f = u_f.at[:, -1].set(u_f[:, 0])
        v_f = _asselin(state.v.data, state.v_before.data, naa.v.data, v_mask3)
        eta_f = _asselin(state.eta.data, state.eta_before.data,
                         naa.eta.data, cmask)
        # Thickness at the three tracer time levels + the Asselin-filtered ssh.
        # e3t_now == h_k (already computed from state.eta above).  e3t_f is built
        # from the filtered eta (eta_f == NEMO r3t_f) so numerator & denominator
        # stay consistent (dynatf/tra_atf use the ssh_atf-filtered scale factor).
        _mwc = self.config.min_water_column_m
        e3t_now = h_k
        e3t_bef = compute_layer_thickness(
            state.eta_before.data, state.H_bathy.data, self.z_coord,
            min_water_column_m=_mwc)
        e3t_aft = compute_layer_thickness(
            naa.eta.data, state.H_bathy.data, self.z_coord,
            min_water_column_m=_mwc)
        e3t_flt = compute_layer_thickness(
            eta_f, state.H_bathy.data, self.z_coord, min_water_column_m=_mwc)
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
        # end-of-step swap (done every step except nit000, which this
        # function's forward-Euler-start branch above handles separately).
        if getattr(self.config, "barotropic_forcing_centred", False):
            naa = naa._replace(
                **_seed_centred_forcing_carry(
                    surface_forcing, freshwater, self.config.rho_0,
                    state.land_mask.data))
        return naa

    def _nemo_mlf_step(self, state: LatLonCGridOceanState, dt: float,
                       freshwater=None, surface_forcing=None, sponge=None,
                       *, grid=None, vertex_mask=None,
                       t_seconds=None,
                       external_tracer_rate=None) -> LatLonCGridOceanState:
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

        ``mlf_baro_corr`` (row 30) — WAIVED per spec §1/§6-5, citing W1a: the
        plan doc (``nemo_faithful_ocean_implementation_plan.md:39-48``)
        certifies this call is a provable no-op for the kamm card specifically
        (no depth-mean source exists under ``surface_stress_implicit=False``,
        the only value this method supports — same construction-time guard
        ``_leapfrog_step`` relies on).  Not built as a real call here (YAGNI —
        zero live callers); a future card enabling ``surface_stress_implicit``
        must add it before claiming this method transcribes that card too.

        Implicit-vmix divisor (spec §6-4, resolved decision 4): this method
        does NOT itself raise on ``config.implicit_vmix_e3t_now_divisor`` (that
        construction-time hard-require is P2's config-surface work) — it simply
        uses whatever the config's divisor is via the SAME ``eta_now=`` kwarg
        ``_leapfrog_step`` already threads to ``_apply_implicit_vertical_
        mixing``.  P2 will add the raise when ``outer_integrator="nemo_mlf"``
        is wired.
        """
        from legoesm.ocean.state import Field
        _grid = grid if grid is not None else self.grid

        # --- FIRST step: forward-Euler start (NEMO l_1st_euler), no RA filter.
        #     Identical to ``_leapfrog_step`` -- see its docstring/comments for
        #     the full NEMO citation (istate.F90:97-99/135-137, stpmlf.F90:
        #     114-117).  A from-rest state's Nbb==Nnn seed makes the leap-frog
        #     combine degenerate to forward-Euler regardless of which method
        #     performs it, so there is nothing MLF-specific to transcribe here.
        if state.u_before is None:
            _entry = state._replace(
                u_before=state.u, v_before=state.v, T_before=state.T,
                S_before=state.S, eta_before=state.eta,
            )
            naa = self._step_impl(
                _entry, dt, freshwater=freshwater,
                surface_forcing=surface_forcing, sponge=sponge, grid=_grid,
                vertex_mask=vertex_mask, t_seconds=t_seconds)
            naa = naa._replace(
                u_before=state.u, v_before=state.v, T_before=state.T,
                S_before=state.S, eta_before=state.eta,
            )
            if getattr(self.config, "barotropic_forcing_centred", False):
                naa = naa._replace(
                    **_seed_centred_forcing_carry(
                        surface_forcing, freshwater, self.config.rho_0,
                        state.land_mask.data))
            return naa

        # --- LEAP-FROG + Asselin.
        rdt = 2.0 * dt
        gamma = self.config.asselin_gamma
        u_mask3 = state.u_mask.data[..., jnp.newaxis]
        v_mask3 = state.v_mask.data[..., jnp.newaxis]
        cmask = state.land_mask.data
        mask3 = cmask[..., jnp.newaxis]
        if isinstance(self.z_coord, OceanPartialCellCoordinate):
            _au3, _av3 = compute_face_masks_3d(self.z_coord.is_active, _grid)
            u_mask3 = u_mask3 * _au3.astype(u_mask3.dtype)
            v_mask3 = v_mask3 * _av3.astype(v_mask3.dtype)
            mask3 = mask3 * self.z_coord.is_active.astype(mask3.dtype)

        # 1. SINGLE tendency pass at Nnn, with dyn_ldf/tra_ldf (+ the isoneutral
        #    Redi tendency, GM/Redi's lego home for row 28) reading Nbb via
        #    ``_ldf_state`` -- replacing ``_leapfrog_step``'s TWO ``_step_impl``
        #    calls (one Nnn-advective, one whole-Nbb-pass-for-diss) with ONE.
        #    "advective" scope still withholds the dissipative increment onto
        #    ``diss_incr`` (same mechanism ``_leapfrog_step`` uses); the
        #    difference is WHAT that increment is evaluated on -- now the Nbb
        #    tracers/velocity read as a LOCAL ARGUMENT by dyn_ldf/tra_ldf only,
        #    not a second whole-state pass (stpmlf.F90:275/437).
        _baro_scale = 2
        state_expl, (K_v_phys, A_v_phys, k33_implicit, surface_tracer_forcing,
                     tke_source, diss_incr, tracer_source) = self._step_impl(
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
        )

        # 2. Explicit combine -- IDENTICAL algebra to ``_leapfrog_step`` (same
        #    baroclinic/barotropic momentum split, same tracer combine, same
        #    thickness_weighted option); ``diss_incr`` now carries the
        #    single-pass Nbb-evaluated dissipation instead of a second pass's.
        h_k = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, self.z_coord,
            min_water_column_m=self.config.min_water_column_m)
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
        _combine = getattr(self.config, "tracer_combine", "concentration")
        if _combine not in ("concentration", "thickness_weighted"):
            raise ValueError(
                f"unknown tracer_combine {_combine!r}; expected "
                '"concentration" or "thickness_weighted"')
        if _combine == "thickness_weighted":
            h_bef = compute_layer_thickness(
                state.eta_before.data, state.H_bathy.data, self.z_coord,
                min_water_column_m=self.config.min_water_column_m)
            h_naa = compute_layer_thickness(
                state_expl.eta.data, state.H_bathy.data, self.z_coord,
                min_water_column_m=self.config.min_water_column_m)
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
        dt_mom = rdt / self.config.dt_mom_ratio
        _tke_prog = self._tke_prognostic_active()
        _tke_old = (state.tke.data if (_tke_prog and state.tke is not None)
                    else None)
        tke_new = None
        if self.config.implicit_vertical_mixing:
            _res = self._apply_implicit_vertical_mixing(
                naa_expl, rdt, surface_forcing,
                K_v_phys=K_v_phys, A_v_phys=A_v_phys, K33_iso=k33_implicit,
                dt_mom=dt_mom,
                surface_tracer_forcing=surface_tracer_forcing,
                tracer_source=tracer_source,
                tke_old=_tke_old, tke_source=tke_source,
                return_tke=_tke_prog, grid=_grid,
                n2_tracers=self._n2_before_advection_tracers(state),
                n2_tracers_before=self._n2_nemo_before_tracers(state),
                eta_now=state.eta.data,
            )
            if _tke_prog:
                naa, tke_new = _res
            else:
                naa = _res
        else:
            naa = naa_expl
        if tke_new is not None:
            if self._tke_advection_active():
                tke_new, _dtke_field = self._apply_tke_advection(
                    state, tke_new, rdt, grid=_grid)
                naa = naa._replace(dtke=_dtke_field)
            naa = naa._replace(
                tke=Field(data=tke_new, name="tke",
                          dims=("lat", "lon", "level"), units="m^2/s^2"))

        # 4. Conservation fixer on the final after-state.
        if self.config.use_conservation_fixer:
            naa = ocean_conservation_fixer(
                naa, state, _grid, self.z_coord, self.config)

        # 5. Robert-Asselin filter -- identical to ``_leapfrog_step``. This IS
        #    row 31's finalize_lbc position relative to rows 32-34 (Asselin):
        #    the continuous masking above already committed every write, so no
        #    extra step is inserted here (see the finalize_lbc docstring note).
        def _asselin(now, before, after, m):
            return (now + gamma * (before - 2.0 * now + after)) * m
        u_f = _asselin(state.u.data, state.u_before.data, naa.u.data, u_mask3)
        u_f = u_f.at[:, -1].set(u_f[:, 0])
        v_f = _asselin(state.v.data, state.v_before.data, naa.v.data, v_mask3)
        eta_f = _asselin(state.eta.data, state.eta_before.data,
                         naa.eta.data, cmask)
        _mwc = self.config.min_water_column_m
        e3t_now = h_k
        e3t_bef = compute_layer_thickness(
            state.eta_before.data, state.H_bathy.data, self.z_coord,
            min_water_column_m=_mwc)
        e3t_aft = compute_layer_thickness(
            naa.eta.data, state.H_bathy.data, self.z_coord,
            min_water_column_m=_mwc)
        e3t_flt = compute_layer_thickness(
            eta_f, state.H_bathy.data, self.z_coord, min_water_column_m=_mwc)
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
        if getattr(self.config, "barotropic_forcing_centred", False):
            naa = naa._replace(
                **_seed_centred_forcing_carry(
                    surface_forcing, freshwater, self.config.rho_0,
                    state.land_mask.data))
        return naa

    def _unsplit_ab2_step(self, state: LatLonCGridOceanState, dt: float,
                          freshwater=None, surface_forcing=None, sponge=None,
                          *, grid=None, vertex_mask=None) -> LatLonCGridOceanState:
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
        g = self.config.g
        eps = self.config.ab2_epsilon
        a_n, a_p = 1.5 + eps, 0.5 + eps
        u_mask = state.u_mask.data
        v_mask = state.v_mask.data
        cmask = state.land_mask.data
        u_mask3 = u_mask[..., jnp.newaxis]
        v_mask3 = v_mask[..., jnp.newaxis]
        mask3 = cmask[..., jnp.newaxis]
        if isinstance(self.z_coord, OceanPartialCellCoordinate):
            _au3, _av3 = compute_face_masks_3d(self.z_coord.is_active, _grid)
            u_mask3 = u_mask3 * _au3.astype(u_mask3.dtype)
            v_mask3 = v_mask3 * _av3.astype(v_mask3.dtype)
            mask3 = mask3 * self.z_coord.is_active.astype(mask3.dtype)

        # 1. Explicit baroclinic tendency (NO surface PGF, NO implicit vmix).
        tend = self.tendencies(state, surface_forcing, sponge=sponge, dt=dt,
                               grid=_grid, vertex_mask=vertex_mask)
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
        if self.config.gm_redi is not None:
            gm_cfg = self.config.gm_redi
            _kri_static, _kri_v_static = static_kappa_redi_override(gm_cfg, _grid)
            _eos_depth = getattr(self.config, "eos_depth", "insitu")
            _gm_dj = None
            if gm_cfg.implicit_K33:
                _gm_dj = gm_redi_density_and_jacobian(
                    state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
                    _grid, self.z_coord, eos=self.config.eos,
                    eos_linear=self.config.eos_linear, mask=cmask,
                    rho_0=self.config.constants.rho_0, g=self.config.constants.g,
                    eos_depth=_eos_depth)
            dT_gm, dS_gm = gm_redi_tracer_tendency_latlon(  # noqa: N806
                state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
                _grid, self.z_coord, gm_cfg, eos=self.config.eos,
                eos_linear=self.config.eos_linear, mask=cmask,
                u_mask=u_mask, v_mask=v_mask,
                rho_0=self.config.constants.rho_0, g=self.config.constants.g,
                omega=self.config.omega,  # see the sibling call's comment above
                kappa_redi_override=_kri_static,
                kappa_redi_v_override=_kri_v_static,
                density_jacobian=_gm_dj, dt=dt, eos_depth=_eos_depth)
            dT_n = dT_n + dt * dT_gm    # noqa: N806
            dS_n = dS_n + dt * dS_gm    # noqa: N806
            if gm_cfg.implicit_K33:
                k33_iso = compute_isoneutral_K33_latlon(
                    state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
                    _grid, self.z_coord, gm_cfg, eos=self.config.eos,
                    eos_linear=self.config.eos_linear, mask=cmask,
                    rho_0=self.config.constants.rho_0, g=self.config.constants.g,
                    kappa_redi_override=_kri_static,
                    kappa_redi_v_override=_kri_v_static,
                    density_jacobian=_gm_dj,
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
            state.eta.data, state.H_bathy.data, self.z_coord,
            min_water_column_m=self.config.min_water_column_m)
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
            n2_tracers=self._n2_before_advection_tracers(state),
            # NEMO e3w(Kmm) divisor (config.implicit_vmix_e3t_now_divisor,
            # #1226 W1): the true pre-barotropic-solve NOW eta (state_corr.eta
            # is the AFTER/Naa level built at step 3 above). No-op when the
            # flag is off (eta_now is read only inside the _e3t_now_slot branch).
            eta_now=state.eta.data)

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

        # barotropic_forcing_centred (#1226 item 3): seed tau_x_prev/
        # tau_y_prev/freshwater_eta_prev from THIS step's forcing (NEMO
        # nit000 rule, sbcmod.F90:568-573 -- "before" set equal to "now" on
        # the very first call, no restart) so a scan driver's first
        # centred step degenerates to plain NOW exactly like the eager
        # ``_leapfrog_step`` forward-Euler-start branch. Reads
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
