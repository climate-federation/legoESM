"""Barotropic solver for the lat-lon C-grid FV ocean model.

Forward-backward substeps for 2D free-surface gravity waves on the
Arakawa C-grid:

    d(eta)/dt = -div(H_total * U_bar, H_total * V_bar)   [cell centers]
    d(U_bar)/dt = f * V_bar_at_u - g * d(eta)/dx          [u-points]
    d(V_bar)/dt = -f * U_bar_at_v - g * d(eta)/dy         [v-points]

The C-grid layout uses compact (single-cell) gradient and divergence
stencils, eliminating the 2*dx checkerboard null space of the A-grid.

Parallels barotropic_latlon.py (A-grid) but with staggered variables.

Wide-halo mode (opt-in, ``BarotropicConfig.barotropic_wide_halo``)
------------------------------------------------------------------
The standard substep loop re-dispatches ~4 latitude halo pads per substep
(``pad_ns_zero`` on the column thickness, the eta-PGF ``gradient_y_cgrid``,
``interp_u_to_vface_4pt``, and the eta-diffusion gradient), i.e. O(4 x
n_substeps) messages per baroclinic step on a band decomposition — a pure
latency term at high rank counts.  The wide-halo path instead exchanges ONE
halo of width ``W = k x r`` (``k`` substeps per exchange, ``r`` = the
per-substep stencil reach) and runs ``k`` substeps communication-free on the
extended band: halo garbage creeps inward at most ``r`` rows per substep, so
after ``k`` substeps the owned rows are still exact.  Serial results are
value-identical (the extended band reproduces the serial neighborhood);
MPI/SPMD parity is gated by tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import LatLonCGridOceanState, LatLonCGridOceanConfig
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    fold_is_local,
    north_fold_mask,
    apply_north_fold,
    divergence_cgrid,
    fold_vface_row,
    gradient_x_cgrid,
    gradient_y_cgrid,
    interp_u_to_vface_4pt,
    min_cell_to_uface,
    min_cell_to_vface,
    pad_ns_zero,
)
from legoesm.grids.halo_latlon import zero_polar_lat_ends as _zero_polar_lat_ends
from legoesm.ocean.dynamics.eta_floor import clamp_and_redistribute as _clamp_redistribute
from legoesm.ocean.dynamics.barotropic_common import (
    bebt_blend,
    compute_filter_weights,
    compute_nemo_boxcar_centred_weights,
    compute_power_law_filter_weights,
    coriolis_at_faces,
    maxvel_clip,
)
from legoesm.ocean.dynamics.ocean_tendency_common import (
    depth_average_to_faces,
)


def _depth_average_to_faces(
    u_3d: jnp.ndarray,
    v_3d: jnp.ndarray,
    h_k: jnp.ndarray,
    min_water_col: jnp.ndarray,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    grid=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute depth-averaged velocities at C-grid face points.

    Parameters
    ----------
    u_3d : (n_lat, n_lon+1, nlev)
    v_3d : (n_lat+1, n_lon, nlev)
    h_k : (n_lat, n_lon, nlev) layer thickness at cell centers.
    min_water_col : scalar
    mask : (n_lat, n_lon)
    u_mask : (n_lat, n_lon+1)
    v_mask : (n_lat+1, n_lon)

    Returns
    -------
    U_bar : (n_lat, n_lon+1)
    V_bar : (n_lat+1, n_lon)
    """
    # h at u/v-faces — min-rule (MOM6/MITgcm hFacW convention).
    # Must match the PE tendency and slow-forcing depth-average which
    # both use min_cell_to_uface/min_cell_to_vface.  Arithmetic mean
    # overestimates face depth at topographic steps, creating a
    # barotropic-baroclinic residual that drives spurious currents.
    h_u = min_cell_to_uface(h_k)
    h_v = min_cell_to_vface(h_k, grid)

    # Barotropic-mean face velocities: thickness-weighted depth average
    # masked by the face mask (#517 item 1: shared depth_average_to_faces;
    # floor = min_water_col, passed verbatim → bit-identical).
    U_bar = depth_average_to_faces(u_3d, h_u, u_mask, min_water_col)
    V_bar = depth_average_to_faces(v_3d, h_v, v_mask, min_water_col)

    return U_bar, V_bar


def _substep_stencil_reach(config: LatLonCGridOceanConfig) -> int:
    """Per-substep latitude stencil reach ``r`` of the FB barotropic substep.

    The number of rows one substep's output can be contaminated by per side
    when its input carries garbage beyond the valid region — the wide-halo
    width budget is ``W = k_substeps x r``.

    Term-by-term (see the substep body):

    * continuity (``pad_ns_zero(H_total)`` min-rule + flux divergence): 1;
    * semi-implicit PGF (``gradient_y_cgrid`` of the blended eta, which
      itself depends on the fresh eta): +1  → base 2;
    * ``interp_u_to_vface_4pt`` of the fresh U (whose eta dependence is
      zonal-only) lands within the same 2;
    * flux-form eta diffusion (``barotropic_diffusion_alpha > 0``): the
      grad+div chain acts on the FRESH eta → its total is 2, within base;
    * divergence damping (``barotropic_div_damp > 0``): grad(div(u,v)) on
      the fresh V (reach 2) adds 2 more (div then grad) → 4 total... the
      conservative budget below charges +1 for the V-side asymmetry too.

    The value is deliberately CONSERVATIVE (a too-small reach corrupts
    owned rows; a too-large reach only wastes halo rows) and is pinned
    mechanically by the NaN-sentinel propagation test
    (``tests/ocean/unit/test_barotropic_wide_halo.py``) — trust the test,
    not this comment, when editing the substep body.
    """
    reach = 3
    if config.barotropic.barotropic_div_damp > 0.0:
        reach += 2
    return reach


def _dissipation_coeffs(config, grid, area, dt_s, dtype, mask):
    """Precompute the substep loop's dissipation coefficient arrays.

    Returns ``(nu_face_u, nu_face_v, diff_u_mask, diff_v_mask,
    div_damp_coeff, div_damp_area_u, div_damp_area_v)`` with ``None``
    entries for statically-disabled terms (the loop gates on ``is not
    None``).  Verbatim extraction from the pre-loop block of
    ``barotropic_substeps_latlon_cgrid`` (values bit-identical).
    """
    nu_face_u = nu_face_v = diff_u_mask = diff_v_mask = None
    div_damp_coeff = div_damp_area_u = div_damp_area_v = None

    # Barotropic diffusion — flux-form with face-centered coefficient.
    # Using div(nu_face * grad(eta)) instead of nu_cell * div(grad(eta))
    # ensures exact volume conservation (divergence theorem: sum of
    # div(F)*area = 0 for any flux F with no-flux BCs).
    # The cell-center form nu_cell * laplacian(eta) is non-conservative
    # when nu_cell varies spatially (area varies as cos(lat) on latlon).
    baro_alpha = jnp.asarray(
        config.barotropic.barotropic_diffusion_alpha, dtype=dtype,
    ) * (dt_s / jnp.asarray(config.barotropic.barotropic_diffusion_dt_ref, dtype=dtype))

    # Precompute face-centered diffusion coefficients (grid geometry only,
    # constant across substeps).
    if config.barotropic.barotropic_diffusion_alpha > 0.0:
        # u-face coefficient: average of adjacent cell areas
        nu_face_u = baro_alpha * 0.5 * (jnp.roll(area, 1, axis=1) + area)
        nu_face_u = jnp.concatenate([nu_face_u, nu_face_u[:, 0:1]], axis=1)
        # v-face coefficient: average of adjacent cell areas.  Cell-pad-first
        # (PR357 Bug-2 pattern): pad the cell AREA so the v-face coefficient
        # at a partition cut averages the neighbour rank's adjacent cell area
        # (MPI halo exchange) rather than zero-padding a rank-local interior
        # average.  Pole rows are zero (wall BC) via zero_polar_lat_ends.
        area_p = pad_ns_zero(area)
        nu_face_v = baro_alpha * 0.5 * (area_p[:-1] + area_p[1:])
        # The north fold row stays zero, matching the pre-existing serial
        # behaviour (flux_y = nu_face_v * grad_y * diff_v_mask is therefore
        # zero across the seam regardless of diff_v_mask).  Enabling
        # fold-seam barotropic diffusion (a non-zero fold-row coefficient)
        # is a physics change that requires the gradient_y_cgrid fold fix
        # (PR358) and a tripolar barotropic-diffusion validation case, so it
        # is deferred to PR358 rather than introduced unvalidated here.
        nu_face_v = _zero_polar_lat_ends(nu_face_v).astype(dtype)
        # Face masks for land boundaries (zero flux at coastlines)
        diff_u_mask = mask * jnp.roll(mask, 1, axis=1)
        diff_u_mask = jnp.concatenate(
            [diff_u_mask, diff_u_mask[:, 0:1]], axis=1,
        )
        # Cell-pad-first (PR357 Bug-2 pattern): pad the cell mask so the
        # v-face mask at a partition cut is the product of the two adjacent
        # cells across the cut (MPI halo exchange) rather than a halo-padded
        # interior face.  pad_ns_zero zero-pads the physical pole on every
        # rank (consistent MPI call count); the seam is overwritten only on
        # the rank that owns it (fold_is_local).
        mask_p = pad_ns_zero(mask)
        diff_v_mask = mask_p[:-1] * mask_p[1:]
        diff_v_mask = _zero_polar_lat_ends(diff_v_mask)
        nmask = north_fold_mask(grid)
        if fold_is_local(grid) or nmask is not None:
            north_dm = mask[-1:] * mask[-1:, grid.fold.perm_T]
            diff_v_mask = apply_north_fold(
                diff_v_mask, north_dm, grid, north_mask=nmask)

    # Divergence damping on barotropic velocity: grad(div(u_bar)).
    # Targets the divergent mode that creates the eta checkerboard,
    # while leaving geostrophic (rotational) flow untouched (#205).
    if config.barotropic.barotropic_div_damp > 0.0:
        div_damp_coeff = jnp.asarray(
            config.barotropic.barotropic_div_damp, dtype=dtype,
        ) * (dt_s / jnp.asarray(config.barotropic.barotropic_diffusion_dt_ref, dtype=dtype))
        # u-face area: average of adjacent cells
        div_damp_area_u = 0.5 * (jnp.roll(area, 1, axis=1) + area)
        div_damp_area_u = jnp.concatenate(
            [div_damp_area_u, div_damp_area_u[:, 0:1]], axis=1,
        )
        # v-face area: average of adjacent cells.  Cell-pad-first so the
        # partition-cut v-face averages the neighbour rank's adjacent cell
        # area (MPI halo); pole rows are zero (wall BC) via
        # zero_polar_lat_ends.
        area_p_dd = pad_ns_zero(area)
        div_damp_area_v = 0.5 * (area_p_dd[:-1] + area_p_dd[1:])
        div_damp_area_v = _zero_polar_lat_ends(div_damp_area_v).astype(dtype)

    return (nu_face_u, nu_face_v, diff_u_mask, diff_v_mask,
            div_damp_coeff, div_damp_area_u, div_damp_area_v)


def _run_substep_loop(
    eta, U_bar, V_bar,
    *,
    dt_s, n_loop, w_filter, w_transport,
    grid, config, g, H_bathy, mask, u_mask, v_mask,
    min_water_col, eta_floor, area,
    F_slow_eta, F_slow_u, F_slow_v,
    f_u, f_v, add_barotropic_coriolis,
    coeffs, local_subcycle_clamp,
    linear_free_surface=False,
):
    """The forward-backward substep loop (verbatim extraction).

    Runs ``n_loop`` substeps from the ``(eta, U_bar, V_bar)`` carry and
    returns the raw final carry
    ``(eta_f, U_bar_f, V_bar_f, Hu_sum_f, Hv_sum_f, eta_sum_f, U_sum_f,
    V_sum_f)`` — callers apply the ``w_total`` normalization and any
    post-loop global redistribute themselves (the wide-halo path must do
    both on the CROPPED owned rows, never on the extended band).

    ``local_subcycle_clamp`` replaces the config flag inside the loop: the
    standard path passes the config value; the wide-halo path FORCES True
    (its per-substep clamp must be allreduce-free — a global redistribute
    over the extended band would double-count the halo overlap).

    Function boundaries are invisible to tracing (Python inlining), so this
    extraction is jaxpr-identical to the previous inline loop.
    """
    (nu_face_u, nu_face_v, diff_u_mask, diff_v_mask,
     div_damp_coeff, div_damp_area_u, div_damp_area_v) = coeffs
    use_div_damp = div_damp_coeff is not None
    use_diffusion = nu_face_u is not None

    # BEBT semi-implicit parameter and MAXVEL clipping
    bebt = config.barotropic.bebt
    _maxvel = config.barotropic.maxvel_barotropic
    use_maxvel = _maxvel > 0.0
    dtype = eta.dtype

    # Accumulators for time-averaged barotropic transport (Phase 2a, issue #102).
    # These accumulate the mass fluxes H*U_bar at each substep so the tracer
    # equation can use transport consistent with the barotropic continuity.
    n_lat = eta.shape[0]
    n_lon = eta.shape[1]
    Hu_sum = jnp.zeros((n_lat, n_lon + 1), dtype=dtype)
    Hv_sum = jnp.zeros((n_lat + 1, n_lon), dtype=dtype)
    eta_sum = jnp.zeros((n_lat, n_lon), dtype=dtype)
    U_sum = jnp.zeros((n_lat, n_lon + 1), dtype=dtype)
    V_sum = jnp.zeros((n_lat + 1, n_lon), dtype=dtype)

    def substep_body(wts_i, carry):
        w_i, w_tr_i = wts_i
        """Single barotropic substep with BEBT, slow forcing, MAXVEL, and cosine filter.

        Parameters
        ----------
        w_i : scalar
            Cosine filter weight for this substep (1.0 for box filter).
        carry : tuple
            (eta, U_bar, V_bar, Hu_sum, Hv_sum, eta_sum, U_sum, V_sum)
        """
        (eta_c, U_bar_c, V_bar_c,
         Hu_sum_c, Hv_sum_c, eta_sum_c, U_sum_c, V_sum_c) = carry

        if linear_free_surface:
            # NEMO key_linssh barotropic continuity: FIXED column depth H
            # (deta/dt = -div(H*U), traadv.F90 r1_hu_0 convention) — eta does
            # not feed back into the transport depth.
            H_total_c = H_bathy * mask
        else:
            H_total_c = jnp.maximum(eta_c + H_bathy, min_water_col) * mask

        # Forward: update eta from continuity (C-grid divergence)
        # Min-rule face depth (consistent with implicit solver and PE
        # tendency).  Arithmetic mean overestimates face depth at
        # topographic steps, creating a transport mismatch.
        H_u = jnp.minimum(jnp.roll(H_total_c, 1, axis=1), H_total_c)
        H_u = jnp.concatenate([H_u, H_u[:, 0:1]], axis=1)
        # Pole rows are zero (wall BC) on regular lat-lon; fold min-rule
        # on tripolar.
        # Cell-pad-first (PR357 Bug-2 pattern): pad the cell column thickness
        # so the v-face min at a partition cut uses the neighbour rank's
        # adjacent column (MPI halo exchange).  Every rank calls pad_ns_zero
        # (consistent MPI call count — the previous direct-concat fold branch
        # skipped it and deadlocked against the else branch); the fold seam is
        # overwritten only on the rank that owns it.
        H_total_pad = pad_ns_zero(H_total_c)
        H_v = jnp.minimum(H_total_pad[:-1], H_total_pad[1:])
        H_v = _zero_polar_lat_ends(H_v)
        nmask = north_fold_mask(grid)
        if fold_is_local(grid) or nmask is not None:
            north = jnp.minimum(
                H_total_c[-1:], fold_vface_row(H_total_c, grid),
            )
            H_v = apply_north_fold(H_v, north, grid, north_mask=nmask)

        flux_u = H_u * U_bar_c * u_mask
        flux_v = H_v * V_bar_c * v_mask

        # Accumulate transport (always box-filtered for volume conservation)
        Hu_sum_new = Hu_sum_c + w_tr_i * flux_u.astype(dtype)
        Hv_sum_new = Hv_sum_c + w_tr_i * flux_v.astype(dtype)

        div_flux = divergence_cgrid(
            flux_u, flux_v, grid, u_mask=u_mask, v_mask=v_mask,
        ).astype(dtype)
        eta_unfloored = (eta_c - dt_s * div_flux + dt_s * F_slow_eta * mask) * mask
        if local_subcycle_clamp:
            # SOTA-local (MOM6/MPAS-O): LOCAL clamp per substep — NO allreduce.
            # The global mass-conserving redistribute is deferred to ONCE per
            # outer step (post-loop, on the time-averaged eta).
            eta_new = jnp.maximum(eta_unfloored, eta_floor) * mask
        else:
            eta_new = _clamp_redistribute(eta_unfloored, eta_floor, mask, area)

        # --- BEBT: Semi-implicit barotropic PGF (#205) ---
        # Blend new and old eta for the pressure gradient to damp fast
        # barotropic gravity waves.  bebt=0 → forward-backward (current),
        # bebt=0.2 → MOM6 default semi-implicit.
        eta_pgf = bebt_blend(eta_new, eta_c, bebt)
        deta_dx = gradient_x_cgrid(eta_pgf, grid).astype(dtype)
        deta_dy = gradient_y_cgrid(eta_pgf, grid).astype(dtype)

        # Average V to u-points for Coriolis
        V_west = jnp.roll(V_bar_c, 1, axis=1)
        V_at_u = 0.25 * (V_bar_c[:-1] + V_bar_c[1:] + V_west[:-1] + V_west[1:])
        V_at_u = jnp.concatenate([V_at_u, V_at_u[:, 0:1]], axis=1)

        # Forward-backward Coriolis (Matsuno) + PGF + slow forcing.  The
        # in-substep Coriolis is gated off when the planetary f×u already
        # reaches the barotropic mode via F_slow (Oceananigans convention) —
        # this removes the C-grid 4-point-average rotational null mode.
        _cor_u = (f_u * V_at_u) if add_barotropic_coriolis else 0.0
        U_bar_new = (U_bar_c + dt_s * (
            _cor_u - g * deta_dx + F_slow_u
        )) * u_mask

        # U averaged to v-points for the backward Coriolis half-step,
        # cell-pad-first (shared interp_u_to_vface_4pt): the partition-
        # cut v-face uses the exact serial 4-point average instead of
        # the neighbour's adjacent FACE row (one row off — the old
        # interior-then-pad_ns_vector_u pattern).  Serial bit-identical;
        # one cell pad per substep replaces one face pad per substep
        # (same collective count on every rank).
        U_new_at_v = interp_u_to_vface_4pt(U_bar_new, grid)
        _cor_v = (-f_v * U_new_at_v) if add_barotropic_coriolis else 0.0
        V_bar_new = (V_bar_c + dt_s * (
            _cor_v - g * deta_dy + F_slow_v
        )) * v_mask

        # Divergence damping: grad(div(u_bar)) (#205)
        if use_div_damp:
            div_uv = divergence_cgrid(
                U_bar_new, V_bar_new, grid,
                u_mask=u_mask, v_mask=v_mask,
            ).astype(dtype)
            grad_div_x = gradient_x_cgrid(div_uv * mask, grid).astype(dtype)
            grad_div_y = gradient_y_cgrid(div_uv * mask, grid).astype(dtype)
            U_bar_new = (
                U_bar_new + div_damp_coeff * div_damp_area_u * grad_div_x
            ) * u_mask
            V_bar_new = (
                V_bar_new + div_damp_coeff * div_damp_area_v * grad_div_y
            ) * v_mask

        # Bottom drag — SINGLE OWNER (finding #6 fix).
        # The 3D PE tendency (``_bc_bottom_drag``) already applies the full
        # bottom drag ``-r·u_bot/h_bot`` (with BBL / partial-cell handling) to
        # ``du_dt``; its depth-mean ``-r·u_bot/H`` is carried into the barotropic
        # mode through ``F_slow_u``/``F_slow_v`` and applied at every substep
        # above.  Re-applying ``implicit_bottom_drag_factor`` here would make the
        # effective barotropic-mode drag ``≈ 2·r/H`` (codex iter-2 finding #1).
        # The drag is therefore owned exclusively by the 3D tendency / F_slow;
        # we do NOT re-apply it here.  (The implicit-CN solver already relied on
        # F_slow alone, so the two barotropic paths are now consistent.)

        # --- MAXVEL clipping: prevent runaway velocities ---
        if use_maxvel:
            U_bar_new = maxvel_clip(U_bar_new, _maxvel)
            V_bar_new = maxvel_clip(V_bar_new, _maxvel)

        # Optional Laplacian damping on eta (flux-form: conservative)
        if use_diffusion:
            grad_x = gradient_x_cgrid(eta_new * mask, grid)
            grad_y = gradient_y_cgrid(eta_new * mask, grid)
            flux_x = nu_face_u * grad_x * diff_u_mask
            flux_y = nu_face_v * grad_y * diff_v_mask
            eta_new = (
                eta_new + divergence_cgrid(flux_x, flux_y, grid).astype(dtype)
            ) * mask
            if local_subcycle_clamp:
                eta_new = jnp.maximum(eta_new, eta_floor) * mask
            else:
                eta_new = _clamp_redistribute(eta_new, eta_floor, mask, area)

        # Accumulate eta, U_bar, V_bar with cosine filter weights
        eta_sum_new = eta_sum_c + w_i * eta_new
        U_sum_new = U_sum_c + w_i * U_bar_new
        V_sum_new = V_sum_c + w_i * V_bar_new

        return (eta_new, U_bar_new, V_bar_new,
                Hu_sum_new, Hv_sum_new, eta_sum_new, U_sum_new, V_sum_new)

    init_carry = (eta, U_bar, V_bar, Hu_sum, Hv_sum, eta_sum, U_sum, V_sum)

    if config.barotropic.differentiable_barotropic:
        # scan path: pass (averaging, transport) weights as xs per substep
        def scan_body(carry, wts_i):
            new_carry = substep_body(wts_i, carry)
            return new_carry, None

        finals, _ = jax.lax.scan(
            scan_body, init_carry, xs=(w_filter, w_transport), length=n_loop,
        )
    else:
        # fori_loop path: index into the filter + transport weights
        def fori_body(i, carry):
            return substep_body((w_filter[i], w_transport[i]), carry)

        finals = jax.lax.fori_loop(0, n_loop, fori_body, init_carry)

    return finals


def _compute_weights(config, n_substeps: int, dtype):
    """Filter + transport weights for the substep loop (shared verbatim).

    Cosine time filter for time-averaging (replaces box-average).
    Cosine-bell (Hanning) window suppresses the side lobes of the box
    filter that alias barotropic modes into the baroclinic coupling.
    Averaging filter for eta/U/V + the matching transport weights for Hu/Hv.
    "power_law" = Shchepetkin-McWilliams (2005) extended-window filter (ROMS/
    MOM6/Oceananigans), which damps the 2Δx barotropic Coriolis null mode the
    first-order cosine filter excites (docs/issues/barotropic_mode_noise.md).
    For box/cosine the transport weights are NO LONGER a flat 1/n: they are the
    continuity-consistent SM2005 tail-sum ``tail_j/(n·w_total)`` returned by
    compute_filter_weights, so the discrete continuity invariant
    ``div(Hu_avg) == (eta_old - eta_avg)/dt`` (which the flux-form tracer step
    needs to preserve a uniform tracer) holds for EVERY filter — the flat 1/n
    broke it for both box (~95%) and cosine (~99%).
    """
    if config.barotropic.barotropic_time_filter == "power_law":
        w_filter, w_total, w_transport, n_loop = compute_power_law_filter_weights(
            n_substeps, dtype,
        )
    elif config.barotropic.barotropic_time_filter == "nemo_boxcar_centred":
        # NEMO dynspg_ts ln_bt_fw=F + nn_bt_flt=1 (forward-frame
        # reduction; see compute_nemo_boxcar_centred_weights).
        w_filter, w_total, w_transport, n_loop = (
            compute_nemo_boxcar_centred_weights(n_substeps, dtype))
    else:
        use_cosine_filter = config.barotropic.barotropic_time_filter == "cosine"
        w_filter, w_total, w_transport = compute_filter_weights(
            n_substeps, dtype, use_cosine=use_cosine_filter,
        )
        n_loop = n_substeps
    return w_filter, w_total, w_transport, n_loop


def barotropic_substeps_latlon_cgrid(
    state: LatLonCGridOceanState,
    dt_s: float,
    n_substeps: int,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig,
    F_slow_eta=None,
    F_slow_u=None,
    F_slow_v=None,
    add_barotropic_coriolis: bool = True,
    t_seconds=None,
) -> LatLonCGridOceanState:
    """Run barotropic substeps on a C-grid lat-lon grid.

    ``add_barotropic_coriolis`` (default True) applies the explicit f×U_bt
    Coriolis term inside each substep.  Set False when the planetary Coriolis
    already reaches the barotropic mode through ``F_slow_u/v`` (its depth-mean,
    via ``coriolis_scheme="explicit_ab2"``) — this is the Oceananigans /
    split-explicit convention (the barotropic equation is ∂_tU = −gH∇η + G^U,
    with NO in-substep Coriolis), and it avoids the C-grid 4-point Coriolis
    rotational null mode (the 2Δx barotropic checkerboard; see
    docs/issues/barotropic_mode_noise.md §A) that otherwise grows under an
    eddy field and blows the eddy-resolving jet.

    Parameters
    ----------
    state : LatLonCGridOceanState
        State after slow tendency application.
    dt_s : float
        Substep size [seconds].
    n_substeps : int
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    config : LatLonCGridOceanConfig
    F_slow_eta : jax.Array or None, shape (n_lat, n_lon)
        Slow forcing for eta (e.g., freshwater mass flux) [m/s].
    F_slow_u : jax.Array or None, shape (n_lat, n_lon+1)
        Depth-averaged slow forcing for u (baroclinic PGF + viscosity
        + advection) [m/s^2].  Passed as constant forcing each substep
        for MOM6-style slow-forcing coupling.
    F_slow_v : jax.Array or None, shape (n_lat+1, n_lon)
        Depth-averaged slow forcing for v [m/s^2].

    Returns
    -------
    LatLonCGridOceanState with updated eta and velocity.
    """
    g = jnp.asarray(config.g)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    u = state.u.data
    v = state.v.data
    eta_raw = state.eta.data
    # Cast all closure-captured arrays to eta's dtype so the fori_loop
    # carry stays in a single precision throughout.  Without this,
    # H_bathy (float64 under x64) promotes H_total → H_u → U_bar_new
    # to float64 while the accumulators (U_sum, V_sum) remain float32,
    # causing a carry-type mismatch in jax.lax.fori_loop.
    _dt = eta_raw.dtype
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=_dt)
    dt_s = jnp.asarray(dt_s, dtype=_dt)
    g = g.astype(_dt)
    H_bathy = H_bathy.astype(_dt)
    mask = mask.astype(_dt)
    u_mask = u_mask.astype(_dt)
    v_mask = v_mask.astype(_dt)
    u = u.astype(_dt)
    v = v.astype(_dt)
    eta_floor = min_water_col - H_bathy
    eta = jnp.maximum(eta_raw, eta_floor) * mask
    _area = grid.area.astype(_dt)

    if F_slow_eta is None:
        F_slow_eta = jnp.zeros_like(eta)
    else:
        F_slow_eta = F_slow_eta.astype(eta.dtype)
    if F_slow_u is None:
        F_slow_u = jnp.zeros((eta.shape[0], eta.shape[1] + 1), dtype=eta.dtype)
    else:
        F_slow_u = F_slow_u.astype(eta.dtype)
    if F_slow_v is None:
        F_slow_v = jnp.zeros((eta.shape[0] + 1, eta.shape[1]), dtype=eta.dtype)
    else:
        F_slow_v = F_slow_v.astype(eta.dtype)

    # --- Equilibrium-tide barotropic body force (OPT-IN; #tidal_forcing) -------
    # Add a = +g*grad(eta_eq_eff) to the SLOW forcing so it (a) is applied at
    # every substep as a constant-over-the-baroclinic-step body force (the tide
    # is slowly varying vs the ~s barotropic subcycle), and (b) is MASKED by
    # u_mask/v_mask together with F_slow inside the substep (lines below:
    # ``(... + F_slow_u) * u_mask``) — so closed/land faces receive nothing.
    # Feature-gated on the STATIC config bool (CLAUDE.md feature-gating exception)
    # AND a supplied traced model time: disabled / no-time => bit-identical.
    _tf_cfg = getattr(config, "tidal_forcing", None)
    if _tf_cfg is not None and _tf_cfg.enabled and t_seconds is not None:
        from legoesm.ocean.physics.tidal_forcing import tidal_acceleration
        _a_tide_x, _a_tide_y = tidal_acceleration(grid, t_seconds, _tf_cfg, g=g)
        F_slow_u = F_slow_u + _a_tide_x.astype(F_slow_u.dtype)
        F_slow_v = F_slow_v + _a_tide_y.astype(F_slow_v.dtype)

    # Depth-averaged velocity.  Cast h_k to _dt because z_coord.sigma_w
    # may be float64 (jnp.linspace default under x64), which would
    # promote U_bar/V_bar and break the fori_loop carry-type invariant.
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    ).astype(_dt)
    U_bar, V_bar = _depth_average_to_faces(
        u, v, h_k, min_water_col, mask, u_mask, v_mask, grid,
    )

    # Semi-implicit Coriolis parameter at face points (#517: shared helper,
    # prefers stored grid.f_u/f_v, fold-safe).
    f_u, f_v = coriolis_at_faces(grid, eta.dtype)

    coeffs = _dissipation_coeffs(config, grid, _area, dt_s, eta.dtype, mask)

    w_filter, w_total, w_transport, n_loop = _compute_weights(
        config, n_substeps, eta.dtype)

    (eta_f, U_bar_f, V_bar_f,
     Hu_sum_f, Hv_sum_f, eta_sum_f, U_sum_f, V_sum_f) = _run_substep_loop(
        eta, U_bar, V_bar,
        dt_s=dt_s, n_loop=n_loop, w_filter=w_filter, w_transport=w_transport,
        grid=grid, config=config, g=g, H_bathy=H_bathy, mask=mask,
        u_mask=u_mask, v_mask=v_mask, min_water_col=min_water_col,
        eta_floor=eta_floor, area=_area,
        F_slow_eta=F_slow_eta, F_slow_u=F_slow_u, F_slow_v=F_slow_v,
        f_u=f_u, f_v=f_v, add_barotropic_coriolis=add_barotropic_coriolis,
        coeffs=coeffs,
        local_subcycle_clamp=config.barotropic.barotropic_local_subcycle_clamp,
        linear_free_surface=getattr(z_coord, 'linear_free_surface', False),
    )

    # Time-averaged barotropic transport: w_transport already carries the full
    # continuity-consistent normalisation — the SM2005 tail-sum
    # ``tail_j/(n·w_total)`` (box/cosine) or the SM2005 secondary weights
    # (power_law) — so the accumulator IS the time-averaged transport ``Hu_avg``
    # that closes ``div(Hu_avg) == (eta_old - eta_avg)/dt``.
    Hu_avg = Hu_sum_f
    Hv_avg = Hv_sum_f

    # Time-averaged eta and velocity (cosine or box filtered)
    eta_avg = eta_sum_f / w_total
    U_bar_avg = U_sum_f / w_total
    V_bar_avg = V_sum_f / w_total

    # SOTA-local split-explicit: the per-substep clamp was LOCAL (no allreduce);
    # restore GLOBAL mass conservation with ONE redistribute call on the
    # time-averaged eta (the returned SSH).  No-op (bit-identical to the
    # per-substep path) when no cell hit eta_floor; this single call's 3 batched
    # allreduces replace the subcycle's ~3*n_substeps (the outer-step
    # fix_eta_drift fixer is separate, unaffected).
    if config.barotropic.barotropic_local_subcycle_clamp:
        eta_avg = _clamp_redistribute(eta_avg, eta_floor, mask, _area)

    # Correct 3D velocities: preserve baroclinic structure.
    # Use time-averaged barotropic velocity for the 3D correction to ensure
    # consistency with eta_avg (the time-averaged eta used for layer thicknesses).
    u_baro_old = U_bar[..., jnp.newaxis]
    v_baro_old = V_bar[..., jnp.newaxis]
    u_prime = u - u_baro_old
    v_prime = v - v_baro_old
    u_new = (u_prime + U_bar_avg[..., jnp.newaxis]) * u_mask[..., jnp.newaxis]
    v_new = (v_prime + V_bar_avg[..., jnp.newaxis]) * v_mask[..., jnp.newaxis]

    state_new = state._replace(
        eta=state.eta.replace(data=eta_avg),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
    return state_new, (Hu_avg, Hv_avg)


# ============================================================================
# Wide-halo split-explicit barotropic (opt-in; scaling-audit item 3)
# ============================================================================

def estimate_barotropic_halo_messages(
    config: LatLonCGridOceanConfig, n_substeps: int,
) -> dict:
    """Analytic per-baroclinic-step lat-halo message counts, standard vs wide.

    The STANDARD substep loop dispatches, per substep: ``pad_ns_zero`` on the
    column thickness, ``gradient_y_cgrid`` on the PGF eta,
    ``interp_u_to_vface_4pt`` on the fresh U (3 pads), +1 pad when the eta
    diffusion is on, +2 when divergence damping is on (its grad_y +
    the diffusion-coeff pads are precomputed once, not counted).  The WIDE
    path replaces them with 2 fused exchanges (cell + v-face groups) for the
    geometry, 2 for the static fields, and 2 per additional chunk.

    Used by the scaling benches to record the communication saving next to
    the timing row (halo-count metric of the wide-halo audit item); counts
    are per interior rank, exchanges not sendrecv pairs.
    """
    pads_per_substep = 3
    if config.barotropic.barotropic_diffusion_alpha > 0.0:
        pads_per_substep += 1
    if config.barotropic.barotropic_div_damp > 0.0:
        pads_per_substep += 2
    _, _, _, n_loop = _compute_weights(config, n_substeps, jnp.float32)
    reach = _substep_stencil_reach(config)
    chunk = int(config.barotropic.barotropic_wide_halo_chunk)
    return {
        "standard_messages": pads_per_substep * int(n_loop),
        "wide_messages_fixed": 4,     # geometry (2) + static fields (2)
        "wide_messages_per_chunk": 2,  # carry (cell + v-face groups)
        "n_loop": int(n_loop),
        "stencil_reach": reach,
        "chunk_config": chunk,
    }


def barotropic_substeps_wide_halo_latlon_cgrid(
    state: LatLonCGridOceanState,
    dt_s: float,
    n_substeps: int,
    grid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig,
    F_slow_eta=None,
    F_slow_u=None,
    F_slow_v=None,
    add_barotropic_coriolis: bool = True,
    t_seconds=None,
) -> LatLonCGridOceanState:
    """Wide-halo twin of :func:`barotropic_substeps_latlon_cgrid`.

    Exchange ONE lat halo of width ``W = k x r`` (``k`` substeps per
    exchange, ``r`` = :func:`_substep_stencil_reach`), then run ``k``
    substeps on the extended band with every operator-internal pad forced
    LOCAL (:func:`legoesm.grids.halo.local_halo_pads`): halo garbage creeps
    inward at most ``r`` rows per substep, so the owned rows stay exact.
    Repeat per chunk; crop to the owned band at the end.

    Contract / scope (v1):

    * regular lat-lon band decompositions only — REFUSES an active tripolar
      fold (the fold row needs a permuted sign-flipped wide exchange) and
      assumes wall-BC poles;
    * per-substep eta clamping runs in the LOCAL (allreduce-free) mode
      regardless of ``barotropic_local_subcycle_clamp`` — a per-substep
      global redistribute over the EXTENDED band would double-count the
      halo overlap.  Global volume conservation is restored by the single
      post-loop redistribute on the CROPPED owned rows (always applied on
      this path), exactly the SOTA-local scheme;
    * uneven band layouts: the auto chunk assumes the neighbour band is at
      least as tall as the halo; with strongly uneven bands (``--wet-balance``)
      set ``barotropic_wide_halo_chunk`` so ``chunk x reach <= min band
      height`` across ranks.

    Numerics: identical update operators via the SAME
    :func:`_run_substep_loop`; serial results match the standard local-clamp
    path to re-association tolerance (parity-gated in
    ``tests/ocean/unit/test_barotropic_wide_halo.py``).
    """
    from legoesm.grids.halo import local_halo_pads
    from legoesm.grids.halo_latlon import (
        widen_band_cell_fields,
        widen_band_vface_fields,
        widen_cgrid_geometry_band,
    )

    if not (hasattr(grid, "dx_u") and hasattr(grid, "fold")):
        raise TypeError(
            "barotropic_substeps_wide_halo_latlon_cgrid requires a "
            "LatLonCGridGeometry (the model's ensure_geometry output); got "
            f"{type(grid).__name__}."
        )

    g = jnp.asarray(config.g)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    u = state.u.data
    v = state.v.data
    eta_raw = state.eta.data
    _dt = eta_raw.dtype
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=_dt)
    dt_s = jnp.asarray(dt_s, dtype=_dt)
    g = g.astype(_dt)
    H_bathy = H_bathy.astype(_dt)
    mask = mask.astype(_dt)
    u_mask = u_mask.astype(_dt)
    v_mask = v_mask.astype(_dt)
    u = u.astype(_dt)
    v = v.astype(_dt)
    eta_floor = min_water_col - H_bathy
    eta = jnp.maximum(eta_raw, eta_floor) * mask

    if F_slow_eta is None:
        F_slow_eta = jnp.zeros_like(eta)
    else:
        F_slow_eta = F_slow_eta.astype(eta.dtype)
    if F_slow_u is None:
        F_slow_u = jnp.zeros((eta.shape[0], eta.shape[1] + 1), dtype=eta.dtype)
    else:
        F_slow_u = F_slow_u.astype(eta.dtype)
    if F_slow_v is None:
        F_slow_v = jnp.zeros((eta.shape[0] + 1, eta.shape[1]), dtype=eta.dtype)
    else:
        F_slow_v = F_slow_v.astype(eta.dtype)

    # Depth average LOCALLY (column-local op): the wide exchange then only
    # moves 2-D fields — never the 3-D (nlev) velocity blocks.
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    ).astype(_dt)
    U_bar, V_bar = _depth_average_to_faces(
        u, v, h_k, min_water_col, mask, u_mask, v_mask, grid,
    )

    w_filter, w_total, w_transport, n_loop = _compute_weights(
        config, n_substeps, eta.dtype)

    # --- Static chunk / width budget --------------------------------------
    # The v-face widening exchanges ``halo = W + 1`` cell rows (the stagger
    # trick), so the halo must satisfy W + 1 <= band height — the exchange
    # pulls rows from ONE neighbour only.
    #
    # RANK-CONSISTENCY (codex: uneven-band deadlock): chunk/W/n_chunks gate
    # COLLECTIVE calls, so they must be identical on every rank.  Under the
    # MPI band layout they derive from the GLOBAL (n_lat_global, n_ranks) —
    # never from the local band height, which differs across ranks when
    # n_lat % n_ranks != 0.  The local band is only ASSERTED to be within
    # the even-split envelope {base, base+1}: every such band is >= base
    # rows, so a base-derived W fits every neighbour; a custom (wet-balance)
    # band outside the envelope aborts loudly BEFORE any exchange (mpirun
    # kills the world on the nonzero exit).  The 2-D pencil is refused:
    # local_halo_pads would also localize its ZONAL exchanges, silently
    # wrapping E/W inside the lon block.
    reach = _substep_stencil_reach(config)
    nl = int(eta.shape[0])
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    band_height_budget = nl
    if get_halo_backend() == "mpi":
        from legoesm.parallel.latlon_mpi import (
            LatLon2DLayout,
            LatLonBandLayout,
        )
        topology = get_mpi_topology()
        if isinstance(topology, LatLon2DLayout):
            raise ValueError(
                "wide-halo barotropic: the 2-D lat-lon pencil layout is not "
                "supported (forcing pads local would also localize the "
                "ZONAL halo exchanges); use the 1-D band layout or disable "
                "barotropic_wide_halo."
            )
        if isinstance(topology, LatLonBandLayout):
            base = int(topology.n_lat_global) // int(topology.n_ranks)
            if nl not in (base, base + 1):
                raise ValueError(
                    f"wide-halo barotropic: local band height {nl} is "
                    f"outside the even-split envelope {{{base}, {base + 1}}} "
                    f"(n_lat_global={topology.n_lat_global}, n_ranks="
                    f"{topology.n_ranks}) — a custom/wet-balanced layout. "
                    f"The wide-halo chunk budget must be rank-consistent, "
                    f"which this path guarantees only for even-split bands; "
                    f"disable barotropic_wide_halo on this layout."
                )
            band_height_budget = base
    w_max = band_height_budget - 1
    chunk_cfg = int(config.barotropic.barotropic_wide_halo_chunk)
    if chunk_cfg > 0:
        chunk = min(chunk_cfg, int(n_loop))
    else:
        chunk = min(int(n_loop), max(w_max // reach, 1))
    W = chunk * reach
    if W > w_max:
        raise ValueError(
            f"wide-halo barotropic: halo width {W} (= chunk {chunk} x reach "
            f"{reach}) exceeds the band-height budget {w_max} (= min band "
            f"height {band_height_budget} - 1, the one-neighbour exchange "
            f"limit); lower barotropic_wide_halo_chunk (or use more "
            f"substeps per exchange only on taller bands)."
        )
    n_chunks = -(-int(n_loop) // chunk)  # ceil

    # --- Extended geometry + static fields (per-step, fused) ---------------
    grid_ext = widen_cgrid_geometry_band(grid, W)
    (H_ext, mask_ext, umask_ext, Fse_ext, Fsu_ext) = widen_band_cell_fields(
        (H_bathy, mask, u_mask, F_slow_eta, F_slow_u), W)
    (vmask_ext, Fsv_ext) = widen_band_vface_fields((v_mask, F_slow_v), W)

    # Tide on the EXTENDED geometry (analytic in the grid — no exchange
    # needed; owned rows are bit-identical to the standard path's values).
    _tf_cfg = getattr(config, "tidal_forcing", None)
    if _tf_cfg is not None and _tf_cfg.enabled and t_seconds is not None:
        from legoesm.ocean.physics.tidal_forcing import tidal_acceleration
        with local_halo_pads():
            _a_tide_x, _a_tide_y = tidal_acceleration(
                grid_ext, t_seconds, _tf_cfg, g=g)
        Fsu_ext = Fsu_ext + _a_tide_x.astype(Fsu_ext.dtype)
        Fsv_ext = Fsv_ext + _a_tide_y.astype(Fsv_ext.dtype)

    eta_floor_ext = min_water_col - H_ext
    area_ext = grid_ext.area.astype(_dt)
    with local_halo_pads():
        f_u_ext, f_v_ext = coriolis_at_faces(grid_ext, eta.dtype)
        coeffs_ext = _dissipation_coeffs(
            config, grid_ext, area_ext, dt_s, eta.dtype, mask_ext)

    # --- Chunked substep loop ----------------------------------------------
    eta_c, U_c, V_c = eta, U_bar, V_bar
    sums = None
    done = 0
    for _ in range(n_chunks):
        k = min(chunk, int(n_loop) - done)
        # Wide exchange of the carry (2 fused messages) with the REAL backend.
        (eta_x, U_x) = widen_band_cell_fields((eta_c, U_c), W)
        (V_x,) = widen_band_vface_fields((V_c,), W)
        wf = jax.lax.slice_in_dim(w_filter, done, done + k)
        wt = jax.lax.slice_in_dim(w_transport, done, done + k)
        with local_halo_pads():
            finals = _run_substep_loop(
                eta_x, U_x, V_x,
                dt_s=dt_s, n_loop=k, w_filter=wf, w_transport=wt,
                grid=grid_ext, config=config, g=g, H_bathy=H_ext,
                mask=mask_ext, u_mask=umask_ext, v_mask=vmask_ext,
                min_water_col=min_water_col, eta_floor=eta_floor_ext,
                area=area_ext,
                F_slow_eta=Fse_ext, F_slow_u=Fsu_ext, F_slow_v=Fsv_ext,
                f_u=f_u_ext, f_v=f_v_ext,
                add_barotropic_coriolis=add_barotropic_coriolis,
                coeffs=coeffs_ext,
                # NEVER a per-substep global redistribute on the extended
                # band (halo overlap would double-count in the allreduce).
                local_subcycle_clamp=True,
                linear_free_surface=getattr(
                    z_coord, "linear_free_surface", False),
            )
        (eta_ext_f, U_ext_f, V_ext_f,
         Hu_k, Hv_k, eta_sum_k, U_sum_k, V_sum_k) = finals
        # Crop the carry back to the owned band for the next exchange.
        eta_c = eta_ext_f[W:W + nl]
        U_c = U_ext_f[W:W + nl]
        V_c = V_ext_f[W:W + nl + 1]
        # Accumulate the (pointwise) sums on OWNED rows across chunks.
        k_sums = (Hu_k[W:W + nl], Hv_k[W:W + nl + 1],
                  eta_sum_k[W:W + nl], U_sum_k[W:W + nl],
                  V_sum_k[W:W + nl + 1])
        sums = k_sums if sums is None else tuple(
            a + b for a, b in zip(sums, k_sums))
        done += k

    Hu_avg, Hv_avg, eta_sum_f, U_sum_f, V_sum_f = sums

    eta_avg = eta_sum_f / w_total
    U_bar_avg = U_sum_f / w_total
    V_bar_avg = V_sum_f / w_total

    # Global mass conservation: ONE redistribute on the OWNED rows with the
    # REAL backend (its global sums are MPI/SPMD-aware) — the wide path's
    # per-substep clamp was local by construction.
    _area = grid.area.astype(_dt)
    eta_avg = _clamp_redistribute(eta_avg, eta_floor, mask, _area)

    # Correct 3D velocities: preserve baroclinic structure (owned rows only).
    u_baro_old = U_bar[..., jnp.newaxis]
    v_baro_old = V_bar[..., jnp.newaxis]
    u_prime = u - u_baro_old
    v_prime = v - v_baro_old
    u_new = (u_prime + U_bar_avg[..., jnp.newaxis]) * u_mask[..., jnp.newaxis]
    v_new = (v_prime + V_bar_avg[..., jnp.newaxis]) * v_mask[..., jnp.newaxis]

    state_new = state._replace(
        eta=state.eta.replace(data=eta_avg),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
    return state_new, (Hu_avg, Hv_avg)
