"""Barotropic solver for the lat-lon C-grid FV ocean model.

Forward-backward substeps for 2D free-surface gravity waves on the
Arakawa C-grid:

    d(eta)/dt = -div(H_total * U_bar, H_total * V_bar)   [cell centers]
    d(U_bar)/dt = f * V_bar_at_u - g * d(eta)/dx          [u-points]
    d(V_bar)/dt = -f * U_bar_at_v - g * d(eta)/dy         [v-points]

The C-grid layout uses compact (single-cell) gradient and divergence
stencils, eliminating the 2*dx checkerboard null space of the A-grid.

Parallels barotropic_latlon.py (A-grid) but with staggered variables.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import LatLonCGridOceanState, LatLonCGridOceanConfig
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    _fold_is_local,
    divergence_cgrid,
    fold_vface_row,
    gradient_x_cgrid,
    gradient_y_cgrid,
    min_cell_to_uface,
    min_cell_to_vface,
    pad_ns_vector_u,
    pad_ns_zero,
)
from legoesm.grids.halo_latlon import zero_polar_lat_ends as _zero_polar_lat_ends
from legoesm.ocean.dynamics.eta_floor import clamp_and_redistribute as _clamp_redistribute
from legoesm.ocean.dynamics.barotropic_common import (
    bebt_blend,
    compute_filter_weights,
    maxvel_clip,
)
from legoesm.ocean.dynamics.ocean_tendency_common import implicit_bottom_drag_factor


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

    # Fuse the per-face thickness + barotropic-mean column reductions —
    # both reduce ``... * h`` over the same level axis.
    _u_pair = jnp.sum(jnp.stack([h_u, u_3d * h_u], axis=-1), axis=-2)
    H_u = jnp.maximum(_u_pair[..., 0], min_water_col)
    U_bar = _u_pair[..., 1] / H_u * u_mask

    _v_pair = jnp.sum(jnp.stack([h_v, v_3d * h_v], axis=-1), axis=-2)
    H_v = jnp.maximum(_v_pair[..., 0], min_water_col)
    V_bar = _v_pair[..., 1] / H_v * v_mask

    return U_bar, V_bar


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
) -> LatLonCGridOceanState:
    """Run barotropic substeps on a C-grid lat-lon grid.

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

    # Depth-averaged velocity.  Cast h_k to _dt because z_coord.sigma_w
    # may be float64 (jnp.linspace default under x64), which would
    # promote U_bar/V_bar and break the fori_loop carry-type invariant.
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    ).astype(_dt)
    U_bar, V_bar = _depth_average_to_faces(
        u, v, h_k, min_water_col, mask, u_mask, v_mask, grid,
    )

    # Semi-implicit Coriolis parameter at face points
    if hasattr(grid, "f_u") and hasattr(grid, "f_v"):
        f_u = grid.f_u.astype(eta.dtype)
        f_v = grid.f_v.astype(eta.dtype)
    else:
        f_cell = grid.f.astype(eta.dtype)
        f_u = 0.5 * (jnp.roll(f_cell, 1, axis=1) + f_cell)
        f_u = jnp.concatenate([f_u, f_u[:, 0:1]], axis=1)
        f_v_interior = 0.5 * (f_cell[:-1] + f_cell[1:])
        f_v = jnp.concatenate([f_cell[0:1], f_v_interior, f_cell[-1:]], axis=0)

    # Barotropic diffusion — flux-form with face-centered coefficient.
    # Using div(nu_face * grad(eta)) instead of nu_cell * div(grad(eta))
    # ensures exact volume conservation (divergence theorem: sum of
    # div(F)*area = 0 for any flux F with no-flux BCs).
    # The cell-center form nu_cell * laplacian(eta) is non-conservative
    # when nu_cell varies spatially (area varies as cos(lat) on latlon).
    baro_alpha = jnp.asarray(
        config.barotropic_diffusion_alpha, dtype=eta.dtype,
    ) * (dt_s / jnp.asarray(config.barotropic_diffusion_dt_ref, dtype=eta.dtype))

    # Precompute face-centered diffusion coefficients (grid geometry only,
    # constant across substeps).
    if config.barotropic_diffusion_alpha > 0.0:
        area = _area  # already cast to _dt above
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
        nu_face_v = _zero_polar_lat_ends(nu_face_v).astype(eta.dtype)
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
        # the rank that owns it (_fold_is_local).
        mask_p = pad_ns_zero(mask)
        diff_v_mask = mask_p[:-1] * mask_p[1:]
        diff_v_mask = _zero_polar_lat_ends(diff_v_mask)
        if _fold_is_local(grid):
            north_dm = mask[-1:] * mask[-1:, grid.fold.perm_T]
            diff_v_mask = jnp.concatenate(
                [diff_v_mask[:-1], north_dm], axis=0,
            )

    # Divergence damping on barotropic velocity: grad(div(u_bar)).
    # Targets the divergent mode that creates the eta checkerboard,
    # while leaving geostrophic (rotational) flow untouched (#205).
    use_div_damp = config.barotropic_div_damp > 0.0
    if use_div_damp:
        div_damp_coeff = jnp.asarray(
            config.barotropic_div_damp, dtype=eta.dtype,
        ) * (dt_s / jnp.asarray(config.barotropic_diffusion_dt_ref, dtype=eta.dtype))
        # u-face area: average of adjacent cells (_area already cast to _dt)
        div_damp_area_u = 0.5 * (jnp.roll(_area, 1, axis=1) + _area)
        div_damp_area_u = jnp.concatenate(
            [div_damp_area_u, div_damp_area_u[:, 0:1]], axis=1,
        )
        # v-face area: average of adjacent cells.  Cell-pad-first so the
        # partition-cut v-face averages the neighbour rank's adjacent cell
        # area (MPI halo); pole rows are zero (wall BC) via
        # zero_polar_lat_ends.
        area_p_dd = pad_ns_zero(_area)
        div_damp_area_v = 0.5 * (area_p_dd[:-1] + area_p_dd[1:])
        div_damp_area_v = _zero_polar_lat_ends(div_damp_area_v).astype(eta.dtype)

    # Cosine time filter for time-averaging (replaces box-average).
    # Cosine-bell (Hanning) window suppresses the side lobes of the box
    # filter that alias barotropic modes into the baroclinic coupling.
    # Transport accumulators (Hu, Hv) MUST remain box-filtered for exact
    # volume conservation with the discrete continuity equation.
    use_cosine_filter = config.barotropic_time_filter == "cosine"
    w_filter, w_total = compute_filter_weights(
        n_substeps, eta.dtype, use_cosine=use_cosine_filter,
    )

    # BEBT semi-implicit parameter and MAXVEL clipping
    bebt = config.bebt
    _maxvel = config.maxvel_barotropic
    use_maxvel = _maxvel > 0.0

    # Accumulators for time-averaged barotropic transport (Phase 2a, issue #102).
    # These accumulate the mass fluxes H*U_bar at each substep so the tracer
    # equation can use transport consistent with the barotropic continuity.
    n_lat = eta.shape[0]
    n_lon = eta.shape[1]
    Hu_sum = jnp.zeros((n_lat, n_lon + 1), dtype=eta.dtype)
    Hv_sum = jnp.zeros((n_lat + 1, n_lon), dtype=eta.dtype)
    eta_sum = jnp.zeros((n_lat, n_lon), dtype=eta.dtype)
    U_sum = jnp.zeros((n_lat, n_lon + 1), dtype=eta.dtype)
    V_sum = jnp.zeros((n_lat + 1, n_lon), dtype=eta.dtype)

    def substep_body(w_i, carry):
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
        if _fold_is_local(grid):
            north = jnp.minimum(
                H_total_c[-1:], fold_vface_row(H_total_c, grid),
            )
            H_v = jnp.concatenate([H_v[:-1], north], axis=0)

        flux_u = H_u * U_bar_c * u_mask
        flux_v = H_v * V_bar_c * v_mask

        # Accumulate transport (always box-filtered for volume conservation)
        Hu_sum_new = Hu_sum_c + flux_u.astype(eta.dtype)
        Hv_sum_new = Hv_sum_c + flux_v.astype(eta.dtype)

        div_flux = divergence_cgrid(
            flux_u, flux_v, grid, u_mask=u_mask, v_mask=v_mask,
        ).astype(eta.dtype)
        eta_unfloored = (eta_c - dt_s * div_flux + dt_s * F_slow_eta * mask) * mask
        eta_new = _clamp_redistribute(eta_unfloored, eta_floor, mask, _area)

        # --- BEBT: Semi-implicit barotropic PGF (#205) ---
        # Blend new and old eta for the pressure gradient to damp fast
        # barotropic gravity waves.  bebt=0 → forward-backward (current),
        # bebt=0.2 → MOM6 default semi-implicit.
        eta_pgf = bebt_blend(eta_new, eta_c, bebt)
        deta_dx = gradient_x_cgrid(eta_pgf, grid).astype(eta.dtype)
        deta_dy = gradient_y_cgrid(eta_pgf, grid).astype(eta.dtype)

        # Average V to u-points for Coriolis
        V_west = jnp.roll(V_bar_c, 1, axis=1)
        V_at_u = 0.25 * (V_bar_c[:-1] + V_bar_c[1:] + V_west[:-1] + V_west[1:])
        V_at_u = jnp.concatenate([V_at_u, V_at_u[:, 0:1]], axis=1)

        # Forward-backward Coriolis (Matsuno) + PGF + slow forcing
        U_bar_new = (U_bar_c + dt_s * (
            f_u * V_at_u - g * deta_dx + F_slow_u
        )) * u_mask

        U_new_at_v_interior = 0.25 * (
            U_bar_new[:-1, :-1] + U_bar_new[:-1, 1:]
            + U_bar_new[1:, :-1] + U_bar_new[1:, 1:]
        )
        U_new_at_v = pad_ns_vector_u(U_new_at_v_interior, grid)
        V_bar_new = (V_bar_c + dt_s * (
            -f_v * U_new_at_v - g * deta_dy + F_slow_v
        )) * v_mask

        # Divergence damping: grad(div(u_bar)) (#205)
        if use_div_damp:
            div_uv = divergence_cgrid(
                U_bar_new, V_bar_new, grid,
                u_mask=u_mask, v_mask=v_mask,
            ).astype(eta.dtype)
            grad_div_x = gradient_x_cgrid(div_uv * mask, grid).astype(eta.dtype)
            grad_div_y = gradient_y_cgrid(div_uv * mask, grid).astype(eta.dtype)
            U_bar_new = (
                U_bar_new + div_damp_coeff * div_damp_area_u * grad_div_x
            ) * u_mask
            V_bar_new = (
                V_bar_new + div_damp_coeff * div_damp_area_v * grad_div_y
            ) * v_mask

        # Bottom drag: -r * U_bar / H_total
        if config.bottom_drag_r > 0:
            U_bar_new = U_bar_new * implicit_bottom_drag_factor(
                dt_s, config.bottom_drag_r, H_u,
            )
            V_bar_new = V_bar_new * implicit_bottom_drag_factor(
                dt_s, config.bottom_drag_r, H_v,
            )

        # --- MAXVEL clipping: prevent runaway velocities ---
        if use_maxvel:
            U_bar_new = maxvel_clip(U_bar_new, _maxvel)
            V_bar_new = maxvel_clip(V_bar_new, _maxvel)

        # Optional Laplacian damping on eta (flux-form: conservative)
        if config.barotropic_diffusion_alpha > 0.0:
            grad_x = gradient_x_cgrid(eta_new * mask, grid)
            grad_y = gradient_y_cgrid(eta_new * mask, grid)
            flux_x = nu_face_u * grad_x * diff_u_mask
            flux_y = nu_face_v * grad_y * diff_v_mask
            eta_new = (
                eta_new + divergence_cgrid(flux_x, flux_y, grid).astype(eta.dtype)
            ) * mask
            eta_new = _clamp_redistribute(eta_new, eta_floor, mask, _area)

        # Accumulate eta, U_bar, V_bar with cosine filter weights
        eta_sum_new = eta_sum_c + w_i * eta_new
        U_sum_new = U_sum_c + w_i * U_bar_new
        V_sum_new = V_sum_c + w_i * V_bar_new

        return (eta_new, U_bar_new, V_bar_new,
                Hu_sum_new, Hv_sum_new, eta_sum_new, U_sum_new, V_sum_new)

    init_carry = (eta, U_bar, V_bar, Hu_sum, Hv_sum, eta_sum, U_sum, V_sum)

    if config.differentiable_barotropic:
        # scan path: pass filter weights as xs for cosine filtering
        def scan_body(carry, w_i):
            new_carry = substep_body(w_i, carry)
            return new_carry, None

        (eta_f, U_bar_f, V_bar_f,
         Hu_sum_f, Hv_sum_f, eta_sum_f, U_sum_f, V_sum_f), _ = jax.lax.scan(
            scan_body, init_carry, xs=w_filter, length=n_substeps,
        )
    else:
        # fori_loop path: index into filter weights
        def fori_body(i, carry):
            w_i = w_filter[i]
            return substep_body(w_i, carry)

        (eta_f, U_bar_f, V_bar_f,
         Hu_sum_f, Hv_sum_f, eta_sum_f, U_sum_f, V_sum_f) = jax.lax.fori_loop(
            0, n_substeps, fori_body, init_carry,
        )

    # Time-averaged barotropic transport (always box-filtered)
    Hu_avg = Hu_sum_f / n_substeps
    Hv_avg = Hv_sum_f / n_substeps

    # Time-averaged eta and velocity (cosine or box filtered)
    eta_avg = eta_sum_f / w_total
    U_bar_avg = U_sum_f / w_total
    V_bar_avg = V_sum_f / w_total

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
