"""FV3_3D iter 583: atmospheric angular momentum diagnostic.

Faithful port of FV3's ``compute_aam`` (fv_dynamics.F90:1264-1307).

Computes per-column mass-integrated atmospheric angular momentum
(AAM) and the total AAM (area-weighted).  In FV3 this drives
``consv_am`` correction and the ``id_aam``/``id_amdt``
diagnostics — for legoESM we expose it as a pure diagnostic.

Reference
---------
FV3 fv_dynamics.F90 compute_aam:

    aam(i,j) = sum_k ( (r²·Ω + r·ua) · dm )
    where r = R · cos(lat); dm = delp / g

For NH (legoESM): dm = rho_full · dz · area  (full mass per
cell).

For PE (legoESM): dm = delp · area / g  (mass per cell from
hybrid pressure).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants


def compute_atmospheric_angular_momentum(
    u_center: jax.Array,
    rho_full: jax.Array,
    grid,
    hc,
) -> tuple[jax.Array, float]:
    """Compute per-column AAM and total mass-weighted AAM.

    Parameters
    ----------
    u_center : jax.Array, shape (6, n, n, nlev)
        Cell-centered zonal wind component in face-local x.
        For NH state this is ``state.u.data`` directly.
        Note: FV3 uses *geographic* u (ua = u_east).  This
        function takes face-local u; rotation to u_east is
        the caller's responsibility for true AAM.  For drift
        monitoring without rotation, the face-local AAM is
        still a useful conserved-ish quantity.
    rho_full : jax.Array, shape (6, n, n, nlev)
        Full density (rho_0 + rho_prime).  In legoESM NH:
        broadcast ``hc.rho_ref[None, None, None, :]`` and add
        ``state.rho_prime.data``.
    grid : CubedSphereGrid
        Provides ``lat``, ``area``, ``radius``.
    hc : HeightCoordinate
        Provides ``dz`` (layer thickness, shape (nlev,)).

    Returns
    -------
    aam_column : jax.Array, shape (6, n, n)
        Per-column mass-integrated AAM [kg·m²/s].
    aam_total : float
        Globally-summed AAM [kg·m²/s].
    """
    R = float(grid.radius)
    cos_lat = jnp.cos(grid.lat)               # (6, n, n)
    r1 = R * cos_lat                          # (6, n, n)
    r2 = r1 * r1
    omega = constants.Omega

    # Per-cell mass: rho * dz * area (kg)
    dz_b = jnp.asarray(hc.dz)[None, None, None, :]  # (1, 1, 1, nlev)
    area_b = grid.area[..., None]                    # (6, n, n, 1)
    dm = rho_full * dz_b * area_b                    # (6, n, n, nlev)

    # AAM per cell: (r²·Ω + r·u) · dm
    r1_b = r1[..., None]                              # (6, n, n, 1)
    r2_b = r2[..., None]
    aam_cell = (r2_b * omega + r1_b * u_center) * dm
    # Column integral
    aam_column = jnp.sum(aam_cell, axis=-1)           # (6, n, n)
    aam_total = float(jnp.sum(aam_column))
    return aam_column, aam_total


def aam_from_nh_state(state, grid, hc) -> tuple[jax.Array, float]:
    """FV3_3D iter 587: AAM from a NonHydrostaticState, using geographic
    u_east (not face-local u).

    Faithful to FV3's compute_aam which uses ua (geographic east wind).
    Rotates the state's face-local (u, v) to (u_east, v_north) via
    grid.angle before computing AAM.

    Parameters
    ----------
    state : NonHydrostaticState
    grid : CubedSphereGrid
    hc : HeightCoordinate

    Returns
    -------
    aam_column, aam_total : as ``compute_atmospheric_angular_momentum``.
    """
    # Rotate face-local (u_face, v_face) to (u_east, v_north).
    # Inverse of rotate_winds_geo_to_grid:
    #   u_east = cos(angle)*u_face - sin(angle)*v_face
    angle = grid.angle[..., None]  # (6, n, n, 1)
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    u_east = cos_a * state.u.data - sin_a * state.v.data
    rho_ref_b = jnp.asarray(hc.rho_ref)[None, None, None, :]
    rho_full = rho_ref_b + state.rho_prime.data
    return compute_atmospheric_angular_momentum(
        u_east, rho_full, grid, hc,
    )


def apply_aam_correction_nh(state_old, state_new, grid, hc):
    """FV3_3D iter 588: FV3 consv_am correction for NH state.

    Faithful port of FV3 ``fv_dynamics.F90:774-794``.  Enforces
    AAM conservation by adding a solid-body-rotation correction
    proportional to the AM drift over the step:

        u0 = -R · amdt / M_fac_total
        u_east_corr(i,j) = u0 · cos(lat(i,j))
        u_face += u_east_corr · cos(angle)
        v_face += -u_east_corr · sin(angle)

    where:
        amdt = AAM(state_new) - AAM(state_old)
        M_fac_total = sum(R²·cos²(lat) · rho·dz·area)

    The correction is uniform in vertical (broadcast across levels).
    Mountain-torque term (FV3 ``zxg``) is NOT subtracted from
    ``amdt`` — for flat-surface adiabatic runs ``amdt`` is the
    pure dycore drift and the correction restores AM exactly.
    For runs with terrain, the user should subtract the
    physical mountain torque before calling this function.

    Returns
    -------
    state_corrected : NonHydrostaticState
        State with u, v adjusted; other fields unchanged.

    Notes
    -----
    Differentiable end-to-end (no Python control flow on traced
    values).  Bit-for-bit identical to state_new when amdt=0
    (no AM drift).
    """
    # Compute AAM at both states (geographic frame)
    _, aam_old = aam_from_nh_state(state_old, grid, hc)
    _, aam_new = aam_from_nh_state(state_new, grid, hc)
    amdt = aam_new - aam_old  # kg·m²/s

    # M_fac_total = sum over all cells of R²·cos²(lat) · column_mass
    cos2_lat = jnp.cos(grid.lat) ** 2                   # (6, n, n)
    dz_b = jnp.asarray(hc.dz)[None, None, None, :]
    rho_ref_b = jnp.asarray(hc.rho_ref)[None, None, None, :]
    rho_full = rho_ref_b + state_new.rho_prime.data
    column_mass = jnp.sum(rho_full * dz_b, axis=-1) * grid.area  # (6, n, n)
    M_fac_total = jnp.sum(
        (grid.radius ** 2) * cos2_lat * column_mass
    )

    # u0 solid-body angular velocity (in u_east units)
    u0 = -grid.radius * amdt / M_fac_total              # scalar [m/s]

    # Build face-local correction from u_east_corr = u0·cos(lat),
    # v_north_corr = 0.
    cos_lat = jnp.cos(grid.lat)                         # (6, n, n)
    u_east_corr = u0 * cos_lat                          # (6, n, n)
    cos_a = jnp.cos(grid.angle)                         # (6, n, n)
    sin_a = jnp.sin(grid.angle)
    delta_u_face_2d = cos_a * u_east_corr               # (6, n, n)
    delta_v_face_2d = -sin_a * u_east_corr              # (6, n, n)
    # Broadcast across levels
    delta_u = delta_u_face_2d[..., None]
    delta_v = delta_v_face_2d[..., None]

    new_u = state_new.u.replace(data=state_new.u.data + delta_u)
    new_v = state_new.v.replace(data=state_new.v.data + delta_v)
    return state_new._replace(u=new_u, v=new_v)


def aam_drift_nh(state_old, state_new, grid, hc) -> float:
    """FV3_3D iter 587: AAM tendency between two NH states.

    Returns amdt = AAM(state_new) - AAM(state_old), faithful to FV3
    fv_dynamics.F90:768 ``amdt = g_sum(te_2d, ...)``.

    A positive value means AM was injected by the dycore.  In
    adiabatic flat-surface runs amdt should be ~0 (the dycore
    should conserve AM exactly if flux-form & no friction).
    Mountain torque (FV3 ``zxg`` term) is NOT included here — for
    runs with terrain, the physical AM tendency includes
    dt·sum(ps·zxg·area) and this function returns only the
    dycore-internal drift.

    Returns
    -------
    amdt : float
        AAM tendency [kg·m²/s].
    """
    _, aam_old = aam_from_nh_state(state_old, grid, hc)
    _, aam_new = aam_from_nh_state(state_new, grid, hc)
    return aam_new - aam_old


# =============================================================================
# PE (hydrostatic) AAM stack — FV3_3D iter 604
# =============================================================================


def aam_from_pe_state(state, grid, coord) -> tuple[jax.Array, float]:
    """FV3_3D iter 604: AAM from a FV3HydrostaticState.

    PE mirror of iter-587's ``aam_from_nh_state``.  Differences:
    - Winds u_d, v_d on D-grid corners (shape (face, n+1, n+1, nlev));
      averaged to cell-center via 4-point average.
    - Column mass from hybrid coord: ``delp = A·p_ref + B·p_s``,
      ``dm = delp · area / g`` (kg per cell).
    - Face-local (u, v) at cell-center → rotated to u_east via
      ``u_east = cos(angle)·u - sin(angle)·v``.

    Faithful to FV3 compute_aam hydrostatic branch.

    Parameters
    ----------
    state : FV3HydrostaticState
    grid : CubedSphereGrid
    coord : HybridSigmaPressureCoordinate

    Returns
    -------
    aam_column, aam_total
    """
    # D-grid → cell-center wind
    u_d = state.u_d.data
    v_d = state.v_d.data
    u_c = 0.25 * (u_d[:, :-1, :-1, :] + u_d[:, 1:, :-1, :]
                  + u_d[:, :-1, 1:, :] + u_d[:, 1:, 1:, :])
    v_c = 0.25 * (v_d[:, :-1, :-1, :] + v_d[:, 1:, :-1, :]
                  + v_d[:, :-1, 1:, :] + v_d[:, 1:, 1:, :])

    # Rotate to u_east
    angle = grid.angle[..., None]
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    u_east = cos_a * u_c - sin_a * v_c                # (6, n, n, nlev)

    # Column mass per cell: delp / g
    p_s = state.p_s.data
    A_h = jnp.asarray(coord.A_half)[None, None, None, :]
    B_h = jnp.asarray(coord.B_half)[None, None, None, :]
    p_half = A_h * coord.p_ref + B_h * p_s[..., None]
    delp = p_half[..., 1:] - p_half[..., :-1]
    dm = delp * grid.area[..., None] / constants.g   # kg per cell

    # AAM per cell: (r²·Ω + r·u_east) · dm
    R = float(grid.radius)
    cos_lat = jnp.cos(grid.lat)
    r1 = R * cos_lat
    r2 = r1 * r1
    aam_cell = (r2[..., None] * constants.Omega
                + r1[..., None] * u_east) * dm
    aam_column = jnp.sum(aam_cell, axis=-1)            # (6, n, n)
    aam_total = float(jnp.sum(aam_column))
    return aam_column, aam_total


def aam_drift_pe(state_old, state_new, grid, coord) -> float:
    """FV3_3D iter 604: PE AAM tendency.

    Mirror of ``aam_drift_nh`` (iter 587) for hydrostatic state.

    Returns
    -------
    amdt : float
        AAM tendency [kg·m²/s].  Mountain torque NOT subtracted.
    """
    _, aam_old = aam_from_pe_state(state_old, grid, coord)
    _, aam_new = aam_from_pe_state(state_new, grid, coord)
    return aam_new - aam_old


def apply_aam_correction_pe(state_old, state_new, grid, coord):
    """FV3_3D iter 604: PE consv_am correction.

    Mirror of ``apply_aam_correction_nh`` (iter 588) for hydrostatic
    state.  Same algorithm (solid-body-rotation correction):

        u0 = -R · amdt / M_fac_total
        u_east_corr = u0 · cos(lat)
        u_face_corr = cos(angle)·u_east_corr   → applied to u_c
        v_face_corr = -sin(angle)·u_east_corr  → applied to v_c

    For PE we apply the SAME correction to D-grid u_d, v_d (the
    correction is a uniform solid-body rotation so it's identical
    at D-grid corners as at cell centers, up to discretization).

    M_fac_total = Σ R²·cos²(lat) · column_mass where column_mass
    uses delp/g (hybrid coord) instead of rho·dz (NH).

    Returns
    -------
    state_corrected : FV3HydrostaticState
        State with u_d, v_d adjusted; other fields unchanged.
    """
    _, aam_target = aam_from_pe_state(state_old, grid, coord)
    _, aam_curr = aam_from_pe_state(state_new, grid, coord)

    # M_fac_total = sum over cells of R²·cos²·column_mass.
    # Used as the analytic Jacobian estimate dAAM/du0.
    p_s = state_new.p_s.data
    A_h = jnp.asarray(coord.A_half)[None, None, None, :]
    B_h = jnp.asarray(coord.B_half)[None, None, None, :]
    p_half = A_h * coord.p_ref + B_h * p_s[..., None]
    delp = p_half[..., 1:] - p_half[..., :-1]
    column_mass_per_cell = jnp.sum(delp, axis=-1) * grid.area / constants.g
    cos2_lat = jnp.cos(grid.lat) ** 2
    M_fac_total = jnp.sum(
        (grid.radius ** 2) * cos2_lat * column_mass_per_cell
    )

    cos_lat = jnp.cos(grid.lat)
    cos_a_c = jnp.cos(grid.angle)
    sin_a_c = jnp.sin(grid.angle)

    state_curr = state_new
    # Newton iteration: D-grid edge-padding introduces a small
    # mismatch between the analytic correction (cell-center linear)
    # and the AAM operator (D-grid avg + rotation).  Newton converges
    # quickly because the correction is nearly linear.
    for _ in range(5):
        amdt = aam_curr - aam_target
        if abs(amdt) < 1e10:  # negligible drift relative to typical scales
            break
        u0 = -grid.radius * amdt / M_fac_total
        delta_u_face_2d = cos_a_c * u0 * cos_lat
        delta_v_face_2d = -sin_a_c * u0 * cos_lat
        du_pad = jnp.pad(delta_u_face_2d, [(0, 0), (0, 1), (0, 1)],
                         mode="edge")
        dv_pad = jnp.pad(delta_v_face_2d, [(0, 0), (0, 1), (0, 1)],
                         mode="edge")
        state_curr = state_curr._replace(
            u_d=state_curr.u_d.replace(
                data=state_curr.u_d.data + du_pad[..., None],
            ),
            v_d=state_curr.v_d.replace(
                data=state_curr.v_d.data + dv_pad[..., None],
            ),
        )
        _, aam_curr = aam_from_pe_state(state_curr, grid, coord)
    return state_curr
